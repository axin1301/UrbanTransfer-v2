"""Posthoc K7-E knowledge->route->prediction statistics, no labels."""
import numpy as np
import torch
import torch.nn.functional as F

def route_response(net,knowledge,prior,mask,expert,mean,std):
    if knowledge.shape[0]!=prior.shape[0] or expert.shape!=prior.shape:
        raise ValueError('K7-E dimension mismatch')
    p=prior.masked_fill(mask,0.)
    if bool((p.sum(-1)<=0).any()):raise ValueError('Fully masked route')
    p=p/p.sum(-1,keepdim=True)
    if net is None:
        candidate=p;gate=torch.zeros(len(p),device=p.device);final=p
    else:
        logits=p.clamp_min(1e-12).log()+net.net(knowledge)
        candidate=F.softmax(logits.masked_fill(mask,-1e4),dim=-1)
        gate=torch.sigmoid(net.strength(knowledge)).flatten()
        final=p*(1-gate[:,None])+candidate*gate[:,None]
        truth,g=net(knowledge,p,mask)
        if not torch.allclose(truth,final,rtol=1e-5,atol=1e-6):
            raise AssertionError('Reconstructed route differs from K7-D')
        if not torch.allclose(g,gate,rtol=1e-5,atol=1e-6):
            raise AssertionError('Reconstructed gate differs')
    return {'prior':p,'candidate':candidate,'final':final,'gate':gate,
            'baseline':mean+std*(p*expert).sum(-1),
            'prediction':mean+std*(final*expert).sum(-1)}

def summarise(knowledge,response,expert,task_ids):
    def a(z):return z.detach().cpu().numpy() if torch.is_tensor(z) else np.asarray(z)
    k=a(knowledge).astype(float);p=a(response['prior']);c=a(response['candidate'])
    w=a(response['final']);g=a(response['gate']);z=a(expert)
    d=a(response['prediction'])-a(response['baseline'])
    if len(k)==0:raise ValueError('Empty group')
    cd=np.abs(c-p).sum(-1);wd=np.abs(w-p).sum(-1)
    if not np.allclose(wd,g*cd,rtol=1e-4,atol=1e-5):
        raise AssertionError('Anchored route identity failed')
    tasks=np.asarray(task_ids);groups=[]
    for task in set(tasks):
        idx=np.flatnonzero(tasks==task)
        if len(idx)>1:groups.append((len(idx),k[idx].std(0).mean(),
                                     w[idx].std(0).mean(),g[idx].std()))
    def weighted(pos):
        return float(sum(v[0]*v[pos] for v in groups)/sum(v[0] for v in groups)) if groups else float('nan')
    corrs=[]
    if g.std()>1e-10 and len(k)>=3:
        for i in range(k.shape[1]):
            if k[:,i].std()>1e-10:corrs.append(abs(float(np.corrcoef(k[:,i],g)[0,1])))
    return {'n':len(k),'knowledge_mean_feature_std':float(k.std(0).mean()),
      'knowledge_varying_features':int((k.std(0)>1e-6).sum()),
      'knowledge_observed_fraction':float(k[:,18:].mean()),
      'knowledge_within_task_std':weighted(1),
      'gate_mean':float(g.mean()),'gate_std':float(g.std()),
      'gate_min':float(g.min()),'gate_max':float(g.max()),
      'gate_within_task_std':weighted(3),
      'gate_knowledge_abs_corr_max':max(corrs) if corrs else float('nan'),
      'gate_knowledge_corr_valid_dimensions':len(corrs),
      'candidate_prior_l1_mean':float(cd.mean()),
      'candidate_prior_l1_p95':float(np.percentile(cd,95)),
      'final_prior_l1_mean':float(wd.mean()),
      'final_prior_l1_p95':float(np.percentile(wd,95)),
      'candidate_expert_mean_std':float(c.std(0).mean()),
      'route_expert_mean_std':float(w.std(0).mean()),
      'route_expert_within_task_std':weighted(2),
      'expert_output_mean_std':float(z.std(0).mean()),
      'prediction_delta_mean':float(d.mean()),
      'prediction_delta_std':float(d.std()),
      'prediction_delta_abs_mean':float(np.abs(d).mean()),
      'prediction_delta_abs_p95':float(np.percentile(np.abs(d),95)),
      'prediction_delta_max_abs':float(np.abs(d).max()),
      'baseline_pred_std':float(a(response['baseline']).std()),
      'new_pred_std':float(a(response['prediction']).std())}

def expert_rows(response,names):
    def a(x):return x.detach().cpu().numpy()
    p=a(response['prior']);c=a(response['candidate']);w=a(response['final'])
    return [{'expert':name,'prior_mean':float(p[:,i].mean()),
      'candidate_mean':float(c[:,i].mean()),'final_mean':float(w[:,i].mean()),
      'candidate_std':float(c[:,i].std()),'final_std':float(w[:,i].std()),
      'final_minus_prior_mean':float((w[:,i]-p[:,i]).mean()),
      'final_minus_prior_abs_mean':float(np.abs(w[:,i]-p[:,i]).mean())}
      for i,name in enumerate(names)]
