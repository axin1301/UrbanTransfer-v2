import unittest
import numpy as np
from urban_transfer.ktrans_k7d import CUES
from urban_transfer.ktrans_knowledge_expert_alignment import (
    build_expert_cue_proxy,alignment_route)
from scripts.audit_ktrans_zero_shot_protocol import audit,classify_checkpoint
from scripts.prepare_ktrans_source_expert_profiles import prepare

def record(value,confidence=3):
    return {'visual_cues':dict.fromkeys(CUES,value),
            'evidence_relevance':dict.fromkeys(CUES,value),
            'evidence_confidence':dict.fromkeys(CUES,confidence)}

class KnowledgeExpertAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.sources=('a','b','c','d')
        self.scores=[{'pseudo_target':t,'source_expert':s,'spatial_pearson':.5}
              for t in self.sources for s in self.sources if t!=s]
        self.text={t:record(2) for t in self.sources}

    def test_source_only_profiles_and_simplex(self):
        profiles=build_expert_cue_proxy(self.sources,self.scores,self.text)
        self.assertEqual(len(profiles),4)
        self.assertTrue(all(v['n_inner_pairs']==3 for v in profiles.values()))
        anchor=np.array([.1,.2,.3,.4])
        route,extra=alignment_route(self.sources,profiles,record(3),anchor)
        self.assertAlmostEqual(route.sum(),1.)
        self.assertTrue((route>=0).all())
        self.assertIsNone(extra['fallback'])
        self.assertFalse(np.allclose(route,anchor))

    def test_missing_confidence_keeps_anchor(self):
        profiles=build_expert_cue_proxy(self.sources,self.scores,self.text)
        anchor=np.array([.1,.2,.3,.4])
        route,extra=alignment_route(self.sources,profiles,record(3,None),anchor)
        np.testing.assert_allclose(route,anchor)
        self.assertEqual(extra['fallback'],'no_confident_visual_evidence')

    def test_outer_or_self_score_disallowed(self):
        with self.assertRaises(ValueError):
            build_expert_cue_proxy(self.sources,self.scores+
                 [{'pseudo_target':'target','source_expert':'a','spatial_pearson':.5}],self.text)
        with self.assertRaises(ValueError):
            build_expert_cue_proxy(self.sources,self.scores+
                 [{'pseudo_target':'a','source_expert':'a','spatial_pearson':.5}],self.text)

    def test_duplicate_pair_disallowed(self):
        with self.assertRaises(ValueError):
            build_expert_cue_proxy(self.sources,self.scores+[self.scores[0]],self.text)

    def test_no_positive_transfer_fails_closed(self):
        data=[{**r,'spatial_pearson':-.1} for r in self.scores]
        with self.assertRaises(ValueError):
            build_expert_cue_proxy(self.sources,data,self.text)

    def test_protocol_best_pt_is_dev_only(self):
        meta={'normalized_experts':True,'expert_bank':True,
              'heldout_task':'target','source_tasks':list(self.sources)}
        self.assertEqual(classify_checkpoint('/tmp/best.pt',meta)['status'],'DEV_ONLY')
        with self.assertRaises(ValueError):
            audit('/tmp/best.pt',meta,strict=True)
        candidate=audit('/tmp/last.pt',meta,strict=True)
        self.assertEqual(candidate['classification'],'CANDIDATE_SOURCE_ONLY')
        self.assertFalse(candidate['strict_protocol_certified'])

    def test_profile_preparation_rejects_outer_scores(self):
        manifest={'source_tasks':list(self.sources),'heldout_task':'target',
                  'k4_checkpoint':'/tmp/best.pt','rows':[]}
        proto={'outer_task':'target','inner_source_tasks':list(self.sources),
               'outer_checkpoint':'/tmp/best.pt'}
        with self.assertRaises(ValueError):
            prepare(manifest,proto,self.scores+
                    [{'pseudo_target':'target','source_expert':'a','spatial_pearson':.2}],
                    {},'model',262144)

if __name__=='__main__':unittest.main()
