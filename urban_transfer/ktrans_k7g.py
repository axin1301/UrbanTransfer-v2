"""K7-G frozen residual decomposition; diagnostic labels are never fitted."""
import numpy as np

def error_decomposition(y,p,baseline=None):
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    p=np.asarray(p,dtype=np.float64).reshape(-1)
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Invalid aligned finite target/prediction')
    e=p-y;bias=float(e.mean());mse=float(np.mean(e**2))
    centered=float(np.mean((e-bias)**2))
    py=float(p.std());yy=float(y.std())
    cov=float(np.mean((p-p.mean())*(y-y.mean())))
    corr=float(cov/(py*yy)) if py>1e-12 and yy>1e-12 else float('nan')
    # Centered squared error: prediction variance + target variance - twice covariance.
    variance_term=py**2+yy**2-2*cov
    if not np.isclose(mse,bias*bias+centered,rtol=1e-9,atol=1e-9):
        raise AssertionError('MSE != bias^2 + centered residual MSE')
    if not np.isclose(centered,variance_term,rtol=1e-9,atol=1e-9):
        raise AssertionError('Covariance error decomposition mismatch')
    # This is a same-set label-fitted oracle; never report as deployable zero-shot metric.
    if py>1e-12:
        slope=cov/(py*py)
        intercept=float(y.mean()-slope*p.mean())
        oracle=float(np.mean((slope*p+intercept-y)**2))
    else:
        slope=float('nan');intercept=float('nan');oracle=float(yy**2)
    result={'n':len(y),'mse':mse,'bias':bias,'bias_squared':bias*bias,
            'centered_residual_mse':centered,'target_std':yy,'pred_mean':float(p.mean()),
            'target_mean':float(y.mean()),'pred_std':py,'pearson':corr,
            'in_sample_oracle_slope':float(slope),
            'in_sample_oracle_intercept':intercept,
            'in_sample_oracle_mse_NOT_GENERALIZATION':oracle}
    if baseline is not None:
        b=np.asarray(baseline,dtype=np.float64).reshape(-1)
        if b.shape!=p.shape or not np.isfinite(b).all():
            raise ValueError('Invalid paired baseline')
        delta=p-b
        delta_mse=float(np.mean((p-y)**2-(b-y)**2))
        linear=float(2*np.mean((b-y)*delta))
        quadratic=float(np.mean(delta**2))
        if not np.isclose(delta_mse,linear+quadratic,rtol=1e-9,atol=1e-9):
            raise AssertionError('Paired squared-error identity mismatch')
        result.update(delta_mse_vs_k5a=delta_mse,
                      paired_linear_error_alignment=linear,
                      paired_quadratic_perturbation=quadratic,
                      prediction_delta_mean=float(delta.mean()),
                      prediction_delta_std=float(delta.std()),
                      prediction_delta_abs_mean=float(np.abs(delta).mean()),
                      delta_bias_squared=(bias*bias-float(np.mean(b-y))**2),
                      delta_centered_mse=(centered-float(np.var(b-y))))
        if not np.isclose(result['delta_bias_squared']+result['delta_centered_mse'],
                          delta_mse,rtol=1e-9,atol=1e-9):
            raise AssertionError('Paired bias/centered decomposition mismatch')
    return result

def direction_summary(source_rows,eval_rows):
    """Signs only: source rows are router TRAIN samples (not independent)."""
    if source_rows['variant']!=eval_rows['variant'] or source_rows['gate_mode']!=eval_rows['gate_mode']:
        raise ValueError('Unpaired variant/mode')
    a=float(source_rows['delta_mse_vs_k5a'])
    b=float(eval_rows['delta_mse_vs_k5a'])
    def sign(v):
        return 'better' if v< -1e-10 else 'worse' if v>1e-10 else 'unchanged'
    return {'variant':source_rows['variant'],'gate_mode':source_rows['gate_mode'],
            'gate':source_rows['gate'],'source_train_delta_mse_IN_SAMPLE':a,
            'target_dev_delta_mse':b,'source_train_direction':sign(a),
            'target_dev_direction':sign(b),
            'direction_matches':sign(a)==sign(b),
            'caution':'Source-train is in-sample for K7-D Router and K4 bank; heldout meta is DEV only.'}
