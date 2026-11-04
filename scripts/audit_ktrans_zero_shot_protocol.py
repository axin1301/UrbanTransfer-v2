"""Fail-closed provenance audit for K-TransMoE outer checkpoint selection.

This audits metadata and explicit checkpoint paths; it does NOT
certify absence of hidden tuning, dataset overlap or independent tests.
Never open/evaluate heldout labels while deciding protocol eligibility.
"""
import argparse,json,sys
from pathlib import Path
import torch

TARGETS=('healthcare_accessibility','mobility_opportunity',
         'social_opportunity','digital_lte_intensity')

def classify_checkpoint(checkpoint,meta,strict=False):
    name=Path(checkpoint).name
    if meta.get('normalized_experts') is not True or meta.get('expert_bank') is not True:
        raise ValueError('Expected normalized source-expert K4 checkpoint')
    if name=='best.pt':
        reason='K4 best.pt checkpoint selected by mean SCUI/UCSI/UCUI meta MSE, including outer heldout labels'
        result={'status':'DEV_ONLY','reason':reason}
    elif name=='last.pt':
        result={'status':'CANDIDATE_SOURCE_ONLY',
                'reason':'last.pt bypasses K4 best-epoch selection; training history and preprocessing still require an independent audit'}
    else:
        result={'status':'UNVERIFIED',
                'reason':'Unrecognized K4 checkpoint filename; epoch-selection provenance unknown'}
    if strict and result['status']!='CANDIDATE_SOURCE_ONLY':
        raise ValueError('Not eligible for strict-protocol candidate: '+result['reason'])
    return result

def audit(checkpoint,meta,k5a=None,manifest=None,strict=False):
    outer=meta['heldout_task']
    sources=list(meta['source_tasks'])
    if outer in sources or len(set(sources))!=len(sources):
        raise ValueError('Outer heldout task present in source-task experts')
    base=classify_checkpoint(checkpoint,meta,strict=strict)
    findings=[{'component':'k4','status':base['status'],'reason':base['reason']}]
    if k5a is not None:
        if Path(k5a['outer_checkpoint']).resolve()!=Path(checkpoint).resolve():
            raise ValueError('K5-A checkpoint is different from audited K4')
        if k5a['outer_task']!=outer or list(k5a['inner_source_tasks'])!=sources:
            raise ValueError('K5-A target/source order mismatch')
        findings.append({'component':'k5a','status':'SOURCE_ONLY_INNER',
             'reason':'K5-A nested pseudo-task scoring uses source tasks; inherited K4 provenance remains '+base['status']})
    if manifest is not None:
        if Path(manifest['k4_checkpoint']).resolve()!=Path(checkpoint).resolve():
            raise ValueError('K7-D manifest K4 checkpoint mismatch')
        if manifest['heldout_task']!=outer or list(manifest['source_tasks'])!=sources:
            raise ValueError('K7-D manifest task mismatch')
        if any(r['task']==outer for r in manifest['rows'] if r['partition']=='train'):
            raise ValueError('Outer task found in router train manifest')
        train_cities={r['city'] for r in manifest['rows'] if r['partition']=='train'}
        unseen_cities={r['city'] for r in manifest['rows'] if r['partition'] in ('meta_UCSI','meta_UCUI')}
        if train_cities & unseen_cities:
            raise ValueError('City overlap between outer train and heldout city rows')
        findings.append({'component':'k7d_manifest','status':'SOURCE_ONLY_ROUTER_ROWS',
            'reason':'Manifest task/city split consistent; rows and caches need further provenance checks'})
    return {'version':'ktrans-zero-shot-provenance-audit-v1','outer_task':outer,
            'source_tasks':sources,'k4_checkpoint':str(Path(checkpoint).resolve()),
            'classification':base['status'],'findings':findings,
            'strict_protocol_certified':False,
            'warning':'No audit of dataset leakage, K4 training history, source-only feature construction, meta-guided research choices, or heldout-task independent evaluation; last.pt is only a candidate.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--k5a-protocol',type=Path)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--require-strict-candidate',action='store_true')
    args=parser.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    meta=saved['meta']
    k5a=json.loads(args.k5a_protocol.read_text()) if args.k5a_protocol else None
    manifest=json.loads(args.manifest.read_text()) if args.manifest else None
    report=audit(args.checkpoint,meta,k5a,manifest,strict=args.require_strict_candidate)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    for item in report['findings']:
        print(f'[protocol-audit] {item["component"]}: {item["status"]} - {item["reason"]}',flush=True)
    print('[protocol-audit] strict_protocol_certified=False',flush=True)
    print('[protocol-audit] saved',args.output,flush=True)
if __name__=='__main__':main()
