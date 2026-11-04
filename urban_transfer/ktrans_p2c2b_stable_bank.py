"""P2-C2B: a single, fixed-identity bank of independent SOURCE Ridge experts.

Diagnostic only, NOT a replacement for the final K4 neural experts.
Each expert is trained exactly once on its own source indicator labels
from the COMMON inner seen-city subset. The input features are frozen
cached RemoteCLIP image embeddings and coordinate/area context; no
learned shared trunk or target-conditioned fitting. A pseudo-task's
own expert is ALWAYS excluded when evaluating that pseudo-task.
"""
import hashlib
from collections import defaultdict
import numpy as np

def model_hash(coef,task,mu,sigma):
    a=np.asarray(coef,dtype='<f8')
    moments=np.asarray([mu,sigma],dtype='<f8')
    return hashlib.sha256(task.encode()+a.tobytes()+moments.tobytes()).hexdigest()

def prepare_records(samples):
    """Expect list of dicts {task,city,image_path,x,y}; no outer target."""
    if not samples:raise ValueError('Empty source records')
    result=[]
    seen=set()
    dim=None
    for s in samples:
        x=np.asarray(s['x'],dtype=float).reshape(-1)
        y=float(s['y'])
        if dim is None:dim=x.size
        if x.size!=dim or not np.isfinite(x).all() or not np.isfinite(y):
            raise ValueError('Invalid source feature or label')
        if not s.get('task') or not s.get('city') or not s.get('image_path'):
            raise ValueError('Source record identity missing')
        key=(s['image_path'],s['task'])
        if key in seen:raise ValueError('Duplicate task/image source record')
        seen.add(key)
        result.append({**s,'x':x,'y':y})
    return result

def fit_bank(samples,source_tasks,heldout_cities,alpha=10.):
    """Fit each SOURCE expert independently, never using another task's labels."""
    if alpha<=0:raise ValueError('Positive Ridge alpha required')
    tasks=tuple(sorted(source_tasks));unseen=set(heldout_cities)
    if len(tasks)<3 or len(tasks)!=len(set(tasks)) or not unseen:
        raise ValueError('Need distinct source tasks and common unseen cities')
    data=prepare_records(samples)
    if set(s['task'] for s in data)!=set(tasks):
        raise ValueError('Non-source or missing training task')
    bank={}
    for task in tasks:
        train=[s for s in data if s['task']==task and s['city'] not in unseen]
        test=[s for s in data if s['task']==task and s['city'] in unseen]
        if len(train)<2 or not test:raise ValueError('Missing source train/eval cities for '+task)
        x=np.vstack([s['x'] for s in train])
        y=np.asarray([s['y'] for s in train])
        mu=float(np.mean(y));std=float(np.std(y))
        sigma=max(std,1e-3)
        z=(y-mu)/sigma
        design=np.column_stack((np.ones(len(x)),x))
        regularizer=np.eye(design.shape[1]);regularizer[0,0]=0.
        # Ordinary positive-penalty ridge, with unregularized intercept.
        coef=np.linalg.solve(design.T@design+alpha*regularizer,design.T@z)
        bank[task]={'coef':coef,'mu':mu,'sigma':sigma,
                    'hash':model_hash(coef,task,mu,sigma),
                    'n_train':len(train),
                    'n_common_unseen':len(test),
                    'train_cities':sorted({s['city'] for s in train})}
    return bank

def predict_bank_row(row,pseudo,source_tasks,bank):
    """Predict four other source experts in a SINGLE frozen bank.

    Pseudo-task semantic scale is the UNIFORM average of available
    SOURCE expert train-only moments, never target labels/moments.
    This fixes a common output scale for all four candidate experts.
    """
    tasks=tuple(sorted(source_tasks))
    if pseudo not in tasks or set(bank)!=set(tasks):
        raise ValueError('Pseudo-task/source bank mismatch')
    x=np.asarray(row['x'],dtype=float).reshape(-1)
    experts=tuple(t for t in tasks if t!=pseudo)
    mean=float(np.mean([bank[t]['mu'] for t in experts]))
    sigma=float(np.mean([bank[t]['sigma'] for t in experts]))
    out={}
    for expert in experts:
        coef=bank[expert]['coef']
        if x.size+1!=len(coef):raise ValueError('Frozen feature dimension mismatch')
        z=float(np.r_[1.,x]@coef)
        out[expert]={'expert_z':z,'prediction_log1p':mean+sigma*z,
                     'semantic_source_mean':mean,'semantic_source_std':sigma,
                     'expert_model_hash':bank[expert]['hash']}
    return out

def export_stable_evidence(old_rows,source_data,bank,source_tasks,unseen):
    """Reuse all ORIGINAL source VLM/labels, replace only expert predictions.

    The row keys, partitions and tile coverage remain directly
    comparable to P2-B1/P2-B2. Outer heldout target never enters.
    """
    tasks=set(source_tasks);unseen=set(unseen)
    index={(r['image_path'],r['task']):r for r in prepare_records(source_data)}
    result=[];counts=defaultdict(int);seen=set()
    for old in old_rows:
        pseudo=old['pseudo_task'];expert=old['expert']
        if pseudo not in tasks or expert not in tasks-{pseudo}:
            raise ValueError('Unexpected old expert or pseudo-target')
        key=(old['image_path'],pseudo)
        src=index.get(key)
        if src is None:raise ValueError('Missing source record for old evidence '+str(key))
        if str(old['city'])!=src['city'] or not np.isclose(float(old['label_log1p']),src['y'],
                                                          rtol=1e-5,atol=1e-5):
            raise ValueError('Old source label/city differs from source training table')
        expected='meta_UCUI' if src['city'] in unseen else 'meta_SCUI'
        if old['inner_partition']!=expected:
            raise ValueError('Existing K5-A fold city split differs from common source split')
        identity=(pseudo,old['inner_partition'],old['tile_id'],expert)
        if identity in seen:raise ValueError('Duplicate source evidence')
        seen.add(identity)
        predicted=predict_bank_row(src,pseudo,source_tasks,bank)[expert]
        error=predicted['prediction_log1p']-src['y']
        result.append({**old,**predicted,'squared_error':float(error*error),
                       'error_log1p':float(error),
                       'scale_source':'uniform_other_source_experts_train_only',
                       'bank_type':'independent_frozen_remoteclip_ridge_v1'})
        counts[pseudo,expected]+=1
    for task in tasks:
        for split in ('meta_SCUI','meta_UCUI'):
            if counts[task,split]<=0:raise ValueError('No evidence for '+task+split)
    return result

def bank_report(bank):
    return [{'expert':name,'n_train':entry['n_train'],
             'n_common_unseen':entry['n_common_unseen'],
             'n_train_cities':len(entry['train_cities']),
             'model_hash':entry['hash'],'source_train_mu':entry['mu'],
             'source_train_std':entry['sigma']} for name,entry in sorted(bank.items())]

def save_bank(path,bank):
    """Persist numeric coefficients and source-train moments, no pickle."""
    tasks=sorted(bank)
    if not tasks:raise ValueError('Empty frozen bank')
    import pathlib
    path=pathlib.Path(path)
    contents={'source_tasks':np.asarray(tasks,dtype='U128')}
    for j,task in enumerate(tasks):
        item=bank[task]
        contents[f'coef_{j}']=np.asarray(item['coef'],dtype=np.float64)
        contents[f'moments_{j}']=np.asarray([item['mu'],item['sigma']],dtype=np.float64)
        contents[f'hash_{j}']=np.asarray(item['hash'],dtype='U64')
    np.savez_compressed(path,**contents)

def load_bank(path):
    """Reload a stable model with full-identity hash verification."""
    with np.load(path,allow_pickle=False) as z:
        tasks=[str(s) for s in z['source_tasks']]
        if len(tasks)!=len(set(tasks)):raise ValueError('Duplicated model names')
        bank={}
        for j,task in enumerate(tasks):
            coef=np.asarray(z[f'coef_{j}'],dtype=float)
            mu,sigma=map(float,z[f'moments_{j}'])
            digest=str(z[f'hash_{j}'])
            if coef.ndim!=1 or coef.size<2 or not np.isfinite(coef).all() or sigma<=0:
                raise ValueError('Invalid frozen model state')
            if digest!=model_hash(coef,task,mu,sigma):
                raise ValueError('Frozen expert state hash mismatch')
            bank[task]={'coef':coef,'mu':mu,'sigma':sigma,'hash':digest}
    return bank
