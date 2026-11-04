import unittest
from scripts.audit_ktrans_k7c_blank import summarize_blank
from scripts.run_ktrans_k7_vlm_pilot import CUES

def sample(field_values):
    return {'scene':'No city scene visible','limitations':'blank',
            'proxy_reasoning':'No image evidence',
            **field_values}

class BlankQualityTests(unittest.TestCase):
    def test_all_null_is_missing_not_zero(self):
        empty={c:None for c in CUES}
        p=sample({'visual_cues':empty,'evidence_relevance':empty,
                  'evidence_confidence':empty})
        rows,review=summarize_blank([
            {'condition':'blank','task':'education_accessibility',
             'tile_id':'a','city':'X','parsed':p}],('education_accessibility',))
        self.assertEqual(rows[0]['visual_cues_nonnull'],0)
        self.assertEqual(rows[0]['evidence_relevance_nonnull'],0)
        self.assertEqual(rows[0]['visual_cues_total'],6)
        self.assertEqual(rows[0]['visual_cues_positive'],0)
        self.assertIsNone(review[0]['human_hallucinated_visible_structure'] or None)

    def test_positive_visual_claim_is_not_automatically_certified(self):
        empty={c:None for c in CUES}
        cue=dict(empty);cue['roads']=3
        conf=dict(empty);conf['roads']=2
        p=sample({'visual_cues':cue,'evidence_relevance':empty,
                  'evidence_confidence':conf})
        rows,_=summarize_blank([
            {'condition':'blank','task':'nighttime_light_intensity',
             'tile_id':'a','city':'X','parsed':p}],('nighttime_light_intensity',))
        self.assertEqual(rows[0]['visual_cues_positive'],1)
        self.assertEqual(rows[0]['evidence_confidence_positive'],1)
        self.assertEqual(rows[0]['blank_visual_fidelity_auto'],'NOT_ASSESSED')

    def test_invalid_json_is_not_counted_as_valid(self):
        rows,review=summarize_blank([
            {'condition':'blank','task':'digital_total_cell_intensity',
             'tile_id':'a','city':'X','parsed':None}],('digital_total_cell_intensity',))
        self.assertEqual(rows[0]['total_records'],1)
        self.assertEqual(rows[0]['valid_json'],0)
        self.assertEqual(rows[0]['visual_cues_total'],0)
        self.assertEqual(len(review),1)

if __name__=='__main__':unittest.main()
