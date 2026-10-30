"""Human-readable metric formatting; never alter persisted numerical precision."""

def metric_text(values):
    """Format scalar metrics to three decimal places, retaining integer counts."""
    parts=[]
    for key in ('n','mse','mae','r2'):
        if key not in values:
            continue
        value=values[key]
        if value is None:
            display='N/A'
        elif key=='n':
            display=str(int(value))
        else:
            display=f'{float(value):.3f}'
        parts.append(f'{key}={display}')
    return ' '.join(parts)

def epoch_text(record):
    train=record['train']['mean_mse']
    val=record['validation']['overall']['mse']
    best=record['best_val_mse']
    return (f"[train] epoch={record['epoch']} "
            f"train_mse={train:.3f} val_mse={val:.3f} "
            f"best_val_mse={best:.3f}")
