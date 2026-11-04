"""K7-H task-cohort mean preservation without indicator labels.

Transductive diagnostic: all samples in a predeclared partition/task
cohort jointly determine the prediction adjustment. Never use y to
compute a shift. The same group membership is applied to every model.
"""
from collections import defaultdict
import numpy as np

def preserve_mean(baseline,prediction,group_ids):
    """Subtract the group mean of prediction-baseline, label-free."""
    base=np.asarray(baseline,dtype=np.float64).reshape(-1)
    pred=np.asarray(prediction,dtype=np.float64).reshape(-1)
    ids=list(group_ids)
    if not len(base) or base.shape!=pred.shape or len(ids)!=len(base):
        raise ValueError('Invalid prediction/group alignment')
    if not np.isfinite(base).all() or not np.isfinite(pred).all():
        raise ValueError('Nonfinite predictions')
    grouped=defaultdict(list)
    for i,group in enumerate(ids):
        if not isinstance(group,str) or not group:
            raise ValueError('Empty/non-string group')
        grouped[group].append(i)
    adjusted=pred.copy()
    audit=[]
    for group in sorted(grouped):
        ix=np.asarray(grouped[group],dtype=int)
        correction=float(np.mean(pred[ix]-base[ix]))
        adjusted[ix]-=correction
        if not np.isclose(adjusted[ix].mean(),base[ix].mean(),atol=1e-10,rtol=1e-10):
            raise AssertionError('Cohort mean preservation failed')
        # A constant offset cannot change within-cohort variance or Pearson.
        if not np.isclose(np.var(adjusted[ix]),np.var(pred[ix]),atol=1e-10,rtol=1e-10):
            raise AssertionError('Cohort variance unexpectedly changed')
        audit.append({'task':group,'n':len(ix),'label_free_shift_subtracted':correction,
                      'original_pred_mean':float(pred[ix].mean()),
                      'baseline_pred_mean':float(base[ix].mean()),
                      'corrected_pred_mean':float(adjusted[ix].mean())})
    return adjusted,audit

def mean_preserving_error_identity(y,base,original,corrected,group_ids):
    """Per cohort, bias² difference is identically zero against baseline.

    A target-label evaluation only, never part of the correction.
    """
    y=np.asarray(y,dtype=np.float64).reshape(-1)
    base=np.asarray(base,dtype=np.float64).reshape(-1)
    original=np.asarray(original,dtype=np.float64).reshape(-1)
    corrected=np.asarray(corrected,dtype=np.float64).reshape(-1)
    if any(len(a)!=len(y) for a in (base,original,corrected)):
        raise ValueError('Inconsistent lengths')
    groups=defaultdict(list)
    for i,g in enumerate(group_ids):groups[g].append(i)
    for ix in groups.values():
        ix=np.asarray(ix,dtype=int)
        if not np.isclose(np.mean(corrected[ix]-y[ix]),np.mean(base[ix]-y[ix]),atol=1e-9):
            raise AssertionError('Label-cohort bias changed')
        centered_original=np.var(original[ix]-y[ix])
        centered_corrected=np.var(corrected[ix]-y[ix])
        if not np.isclose(centered_original,centered_corrected,atol=1e-9):
            raise AssertionError('Centered residual changed after constant shift')
    return True
