import unittest
import torch
from scripts.run_ktrans_k7c_grounding import decode_batch,is_oom

class FakeProcessor:
    def batch_decode(self,tokens,skip_special_tokens=True):
        self.tokens=tokens.clone()
        return ['one  ','two  ']

class K7CBatchTests(unittest.TestCase):
    def test_batched_decode_removes_padded_prompt_width(self):
        processor=FakeProcessor()
        # Two left-padded prompts with the same padded length=4,
        # followed by three new tokens per sample.
        input_ids=torch.tensor([[0,0,11,12],[21,22,23,24]])
        generated=torch.tensor([[0,0,11,12,101,102,103],
                                [21,22,23,24,201,202,203]])
        texts=decode_batch(processor,generated,input_ids)
        self.assertEqual(texts,['one','two'])
        self.assertEqual(processor.tokens.tolist(),[[101,102,103],[201,202,203]])

    def test_batch_alignment_check(self):
        with self.assertRaises(ValueError):
            decode_batch(FakeProcessor(),torch.zeros(3,7,dtype=torch.long),
                         torch.zeros(2,4,dtype=torch.long))

    def test_oom_vs_other_runtime_error(self):
        self.assertTrue(is_oom(RuntimeError('HIP out of memory')))
        self.assertTrue(is_oom(RuntimeError('HIP memory allocation failed')))
        self.assertFalse(is_oom(RuntimeError('bad image decode')))

if __name__=='__main__':unittest.main()
