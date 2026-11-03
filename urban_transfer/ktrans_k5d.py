"""K5-D measurement-aware task-scale inference, zero target-label calibration.

The only fitted quantity is ONE scalar mixture between semantic and
measurement-aware source-scale estimates, using source-only LOTO moments.
No target evaluation labels or target moments are ever used by predict().
"""
import numpy as np
import torch
import torch.nn.functional as F
from .measurement_knowledge import relatedness

def moment_estimate(target,sources,vectors,stats,temperature=.15,
                    measurement=False,domain_bonus=.7,mechanism_bonus=.5):
    if target in sources:raise ValueError('Target must be absent from source pool')
    if not sources or temperature<=0:raise ValueError('Need source tasks and positive temperature')
    t=F.normalize(vectors[target].float(),dim=0)
    s=F.normalize(torch.stack([vectors[k].float() for k in sources]),dim=-1)
    cosine=(s@t).cpu().numpy().astype(float)
    logits=cosine/temperature
    if measurement:
        for i,name in enumerate(sources):
            domain,mechanism=relatedness(target,name)
            logits[i]+=domain_bonus*domain+mechanism_bonus*mechanism
    weights=np.exp(logits-logits.max());weights/=weights.sum()
    mean=np.array([stats[k][0] for k in sources],dtype=float)
    std=np.array([stats[k][1] for k in sources],dtype=float)
    if np.any(std<=0):raise ValueError('Source std must be positive')
    return float(weights@mean),float(np.log(weights@std)),weights

class MeasurementAwareScaleRouter:
    def __init__(self,vectors,source_stats,temperature=.15,shrinkage=1.):
        if shrinkage<=0:raise ValueError('Shrinkage must be positive')
        self.vectors=vectors
        self.stats={k:(float(v[0]),float(v[1])) for k,v in source_stats.items()}
        self.sources=tuple(sorted(self.stats))
        if len(self.sources)<3:raise ValueError('At least 3 source tasks')
        self.temperature=temperature
        # Source-only leave-one-task-out risk for a single mixing coefficient.
        # Per-output normalization prevents the high-scale population task
        # from automatically dominating the regression objective.
        target_mu=np.array([self.stats[k][0] for k in self.sources])
        target_ls=np.log([self.stats[k][1] for k in self.sources])
        denom=np.array([max(float(np.std(target_mu)),.1),
                        max(float(np.std(target_ls)),.1)])
        deltas=[];errors=[]
        for target in self.sources:
            pool=[s for s in self.sources if s!=target]
            a=np.asarray(moment_estimate(target,pool,vectors,self.stats,temperature,False)[:2])
            b=np.asarray(moment_estimate(target,pool,vectors,self.stats,temperature,True)[:2])
            truth=np.array([self.stats[target][0],np.log(self.stats[target][1])])
            deltas.extend(((b-a)/denom).tolist())
            errors.extend(((truth-a)/denom).tolist())
        d=np.asarray(deltas);e=np.asarray(errors)
        self.alpha=float(np.clip(np.dot(d,e)/(np.dot(d,d)+shrinkage),0.,1.))
        self.shrinkage=shrinkage

    def predict(self,target):
        if target in self.stats:raise ValueError('Cannot predict fitted source target')
        base=np.asarray(moment_estimate(target,self.sources,self.vectors,
                                        self.stats,self.temperature,False)[:2])
        aware=np.asarray(moment_estimate(target,self.sources,self.vectors,
                                         self.stats,self.temperature,True)[:2])
        value=(1.-self.alpha)*base+self.alpha*aware
        return float(value[0]),float(np.exp(np.clip(value[1],-7,7)))

    def components(self,target):
        base=moment_estimate(target,self.sources,self.vectors,self.stats,
                             self.temperature,False)
        aware=moment_estimate(target,self.sources,self.vectors,self.stats,
                              self.temperature,True)
        return {'semantic_mean':base[0],'semantic_std':float(np.exp(base[1])),
                'measurement_mean':aware[0],'measurement_std':float(np.exp(aware[1])),
                'semantic_weights':base[2].tolist(),
                'measurement_weights':aware[2].tolist(),'alpha':self.alpha}
