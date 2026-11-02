"""K5-C: source-task-only, task-knowledge-guided scale prediction.

A transparent small-sample leave-one-source-task-out ridge calibrator.
Each training task's moments are excluded from its input features, and
outer heldout task labels/moments are never read for fitting. This is a
pilot scale head, NOT an unbiased target-domain calibration estimate.
"""
import numpy as np
import torch
import torch.nn.functional as F

def scale_features(target,sources,vectors,stats,temperature=.15):
    """Features from OTHER source tasks only: semantic and uniform moments."""
    if target in sources:raise ValueError('Target task must be excluded from source pool')
    if not sources:raise ValueError('Empty source pool')
    target_vec=F.normalize(vectors[target].float(),dim=-1)
    src_vec=F.normalize(torch.stack([vectors[s].float() for s in sources]),dim=-1)
    sims=(src_vec@target_vec).cpu().numpy().astype(float)
    w=np.exp((sims-sims.max())/temperature);w=w/w.sum()
    mu=np.asarray([stats[s][0] for s in sources],dtype=float)
    std=np.asarray([stats[s][1] for s in sources],dtype=float)
    # Four highly interpretable scale predictors; no target moments allowed.
    return np.array([1.,w@mu,mu.mean(),np.log(w@std),np.log(std.mean())],
                    dtype=np.float64)

class IndependentScaleRouter:
    def __init__(self,vectors,source_stats,penalty=5.,temperature=.15):
        if penalty<=0 or temperature<=0:raise ValueError('Positive penalty/temperature required')
        self.vectors=vectors
        self.stats={str(k):(float(v[0]),float(v[1])) for k,v in source_stats.items()}
        self.sources=tuple(sorted(self.stats))
        if len(self.sources)<3:raise ValueError('At least three source tasks required')
        if not set(self.sources).issubset(vectors):raise ValueError('Missing source embeddings')
        self.penalty=penalty;self.temperature=temperature
        xs=[];ys=[]
        for target in self.sources:
            other=[s for s in self.sources if s!=target]
            xs.append(scale_features(target,other,vectors,self.stats,temperature))
            mu,std=self.stats[target]
            ys.append([mu,np.log(std)])
        x=np.asarray(xs);y=np.asarray(ys)
        # Calibrate the zero-shot semantic estimates with small residual
        # corrections. Ridge shrinks toward the frozen semantic baseline,
        # not toward arbitrary task mean; avoid n=5 unregularized fitting.
        base=np.stack([x[:,1],x[:,3]],axis=1)
        residual=y-base
        # Fixed feature scaling, no target metadata used.
        design=np.stack([np.ones(len(x)),x[:,2]-x[:,1],x[:,4]-x[:,3]],axis=1)
        gram=design.T@design+np.diag([penalty,penalty,penalty])
        self.coef=np.linalg.solve(gram,design.T@residual)

    def predict(self,target):
        if target in self.stats:
            raise ValueError('Heldout target must not be a fitted source task')
        x=scale_features(target,self.sources,self.vectors,self.stats,self.temperature)
        offsets=np.array([1.,x[2]-x[1],x[4]-x[3]])@self.coef
        mean=float(x[1]+offsets[0])
        std=float(np.exp(np.clip(x[3]+offsets[1],-7,7)))
        return mean,std

    def semantic_scale(self,target):
        x=scale_features(target,self.sources,self.vectors,self.stats,self.temperature)
        return float(x[1]),float(np.exp(x[3]))
