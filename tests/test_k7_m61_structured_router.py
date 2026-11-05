"""CPU-only M6.1 checks for token segmentation, no-label training gradients."""
import unittest

import torch
import torch.nn.functional as F

from urban_transfer.k7_m61_structured import (
    START_DEF,START_REL,END_REL,find_span,evenly_sample,
    extract_structured,structured_prompt,decoded_marker_spans)
from urban_transfer.k7_m61_router import StructuredMLLMRouter


class FakeTokenizer:
    def encode(self,text,add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):
        return "".join("<|image_pad|>" if x==999 else "\\n[" if x==5000 else chr(x) for x in ids)


class M61Tests(unittest.TestCase):
    def test_structured_prompt_and_no_labels(self):
        prompt=structured_prompt("education_accessibility")
        self.assertIn(START_DEF,prompt)
        self.assertIn(START_REL,prompt)
        self.assertIn(END_REL,prompt)
        self.assertIn("population_count",prompt)
        self.assertIn("food_accessibility",prompt)
        self.assertNotIn("indicator_value_log1p",prompt)

    def test_image_definition_relation_segments(self):
        tokenizer=FakeTokenizer()
        img=[999]*100
        text=(START_DEF+"housing details"+START_REL+
              "roads and access"+END_REL+"final instruction")
        ids=img+tokenizer.encode(text)
        states=torch.randn(len(ids),32)
        out=extract_structured(states,ids,999,tokenizer,(6,4,5))
        self.assertEqual(tuple(out.shape),(15,32))
        self.assertTrue(torch.equal(out[0],states[0]))
        self.assertTrue(torch.equal(out[5],states[99]))
        self.assertTrue(torch.equal(out[6],states[100+len(START_DEF)]))
        self.assertTrue(torch.equal(out[10],states[
            100+len(START_DEF)+len("housing details")+len(START_REL)]))

    def test_contextual_marker_tokenization_is_not_standalone(self):
        # A preceding newline merges with the first '[': standalone
        # marker IDs cannot match this actual processor input sequence.
        tokenizer=FakeTokenizer()
        def merged(marker):
            return [5000]+[ord(ch) for ch in marker[1:]]
        ids=([999]*3+merged(START_DEF)+[ord(c) for c in "Housing"]
             +merged(START_REL)+[ord(c) for c in "Road access"]
             +merged(END_REL)+[ord(c) for c in "End"])
        self.assertNotIn(tokenizer.encode(START_DEF),
                         [ids[i:i+len(START_DEF)] for i in range(len(ids))])
        spans=decoded_marker_spans(ids,tokenizer)
        self.assertEqual(len(spans),2)
        self.assertEqual("".join(chr(ids[i]) for i in spans[0]),"Housing")
        self.assertEqual("".join(chr(ids[i]) for i in spans[1]),"Road access")
        out=extract_structured(torch.randn(len(ids),32),ids,999,tokenizer,(3,4,4))
        self.assertEqual(tuple(out.shape),(11,32))

    def test_ambiguous_markers_fail(self):
        with self.assertRaises(ValueError):
            find_span([1,2,1,2],[1,2])
        with self.assertRaises(ValueError):
            evenly_sample(torch.randn(4,5),[],2)

    def test_unmasked_source_route_gets_supervised_gradient(self):
        torch.manual_seed(42)
        model=StructuredMLLMRouter(12,32,hidden=32,group_counts=(6,4,5))
        images=torch.randn(8,12)
        context=torch.randn(8,4)
        ml=torch.randn(8,15,32)
        labels=torch.randn(8)
        target_tokens=torch.randn(8,15,32)
        main,weights,knowledge=model(images,context,ml,return_aux=True)
        self.assertEqual(tuple(weights.shape),(8,2))
        self.assertTrue(bool((weights>0).all()))
        self.assertTrue(bool((weights<1).all()))
        source_predictions=model.source_expert_predictions(images,context,ml)
        src_idx=torch.tensor([0,1,0,1,0,1,0,1]).unsqueeze(1)
        direct=source_predictions.gather(1,src_idx).squeeze(1)
        target_knowledge=model.read_hidden(target_tokens)
        align=model.text_image_alignment(target_knowledge,images)
        loss=F.mse_loss(main,labels)+.3*F.mse_loss(direct,labels)+.25*align
        loss.backward()
        self.assertGreater(float(model.router[-1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.attn.in_proj_weight.grad.abs().sum()),0)
        self.assertGreater(float(model.align.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.experts[0][1].weight.grad.abs().sum()),0)
        self.assertGreater(float(model.experts[1][1].weight.grad.abs().sum()),0)

    def test_mllm_hidden_mismatch_changes_predictions(self):
        torch.manual_seed(42)
        model=StructuredMLLMRouter(12,32,hidden=32,group_counts=(6,4,5)).eval()
        images=torch.randn(5,12)
        ctx=torch.randn(5,4)
        tok=torch.randn(5,15,32)
        a=model(images,ctx,tok)
        b=model(images,ctx,tok.roll(1,0))
        self.assertGreater(float((a-b).abs().max()),1e-8)


if __name__=="__main__":
    unittest.main()
