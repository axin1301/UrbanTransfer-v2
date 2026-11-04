import json
import tempfile
import unittest
from pathlib import Path
import torch
from scripts.prepare_ktrans_k7d import choose
from scripts.cache_ktrans_k7d import unique_jobs,get_pending
from scripts.run_ktrans_k7d import index_knowledge
from scripts.run_ktrans_k7b_controls import SYSTEM,TEXT_ONLY_SYSTEM
from urban_transfer.ktrans_k7d import (
    CUES,encode_record,knowledge_vector,AnchoredKnowledgeRouter,VARIANTS)

def parsed(kind):
    score={c:2 for c in CUES}
    none={c:None for c in CUES}
    return {'visual_cues':none if kind=='text_only' else score,
            'evidence_relevance':none if kind=='image_only' else score,
            'evidence_confidence':none if kind=='text_only' else score}

class K7DTransferTests(unittest.TestCase):
    def test_null_masks_distinct_from_ordinal_zero(self):
        obj=parsed('image_task')
        obj['visual_cues']['roads']=None
        obj['visual_cues']['built_up']=0
        x=encode_record(obj,'image_task')
        self.assertEqual(x.shape,(36,))
        self.assertEqual(x[0],0.)
        self.assertEqual(x[1],0.)
        self.assertEqual(x[18],1.)  # observed zero
        self.assertEqual(x[19],0.)  # missing value

    def test_three_ablation_features(self):
        it,io,tx=parsed('image_task'),parsed('image_only'),parsed('text_only')
        for variant in VARIANTS:
            x=knowledge_vector(it,io,tx,variant)
            self.assertEqual(x.shape,(36,))
        self.assertEqual(knowledge_vector(it,io,tx,'k5a').abs().sum(),0.)
        self.assertEqual(knowledge_vector(it,io,tx,'text_only')[:6].abs().sum(),0.)
        self.assertEqual(knowledge_vector(it,io,tx,'image_only')[6:12].abs().sum(),0.)

    def test_text_only_cannot_claim_visible_cues(self):
        x=parsed('text_only')
        x['visual_cues']['roads']=1
        with self.assertRaises(ValueError):
            encode_record(x,'text_only')

    def test_router_masks_own_source_expert(self):
        net=AnchoredKnowledgeRouter(5)
        feats=torch.zeros(3,36)
        prior=torch.tensor([[0.,.1,.2,.3,.4],
                            [.2,0.,.2,.2,.4],
                            [.2,.1,0.,.3,.4]],dtype=torch.float32)
        mask=torch.eye(5,dtype=torch.bool)[:3]
        w,g=net(feats,prior,mask)
        self.assertTrue(torch.allclose(w.sum(-1),torch.ones(3),atol=1e-6))
        self.assertLess(w[0,0].item(),1e-9)
        self.assertLess(w[1,1].item(),1e-9)
        self.assertLess(w[2,2].item(),1e-9)
        self.assertTrue((g<.5).all())

    def test_missing_cache_fails_closed(self):
        manifest={'rows':[{'tile_id':'id','image_path':'/tmp/tile.png',
                 'task':'education_accessibility'}]}
        jobs=unique_jobs(manifest,'test',262144)
        self.assertEqual(len(jobs),3)
        self.assertEqual(len(get_pending(jobs,{})),3)
        with self.assertRaises(ValueError):
            index_knowledge(jobs,{})

    def test_cache_only_accepts_full_prompt_identity(self):
        manifest={'rows':[{'tile_id':'id','image_path':'/tmp/tile.png',
                 'task':'education_accessibility'}]}
        jobs=unique_jobs(manifest,'test',262144)
        cache={j['key']:{**j,'parsed':parsed(j['condition'])} for j in jobs}
        result=index_knowledge(jobs,cache)
        self.assertEqual(len(result),3)
        damaged=dict(cache)
        key=jobs[1]['key']
        damaged[key]={**damaged[key],'prompt':'different prompt'}
        with self.assertRaises(ValueError):
            index_knowledge(jobs,damaged)

if __name__=='__main__':unittest.main()
