"""M6.1: structured image/definition/relation token extraction from frozen Qwen2.5-VL.

Token chunks remain in the original MLLM 2048-D space. The visual
chunk is sampled from actual Qwen image-placeholder token positions.
Text spans are located by explicit boundary marker subsequences; fail
rather than silently using the wrong tokens.
"""
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from urban_transfer.k7_mllm_hidden import (
    FrozenMLLM, HiddenStore, TASKS, TRAIN_TASKS, PROFILES, relation_text)

VERSION="mllm-structured-v1"
START_DEF="[URBAN_INDICATOR_DEFINITION_START]"
START_REL="[URBAN_TRANSFER_KNOWLEDGE_START]"
END_REL="[URBAN_TRANSFER_KNOWLEDGE_END]"

def structured_prompt(task):
    if task not in TASKS:
        raise ValueError("Unregistered pilot task")
    definition,measurement,_=PROFILES[task]
    edges=[relation_text(source,task) for source in TRAIN_TASKS if source!=task]
    return (
        "Study this satellite image as visual evidence, without inventing facility counts or numerical labels.\n"
        + START_DEF + "\n"
        + "Target: " + task + "\nDefinition: " + definition
        + "\nMeasurement: " + measurement + "\n"
        + START_REL + "\n"
        + "\n".join(edges) + "\n"
        + END_REL + "\n"
        + "Consider the visible evidence for this target, and the limits of transfer."
    )


def structured_key(path,task,model_id,max_pixels,group_counts):
    p=Path(path).resolve()
    st=p.stat()
    value=[VERSION,str(p),st.st_size,st.st_mtime_ns,task,
           structured_prompt(task),model_id,max_pixels,list(group_counts)]
    return hashlib.sha256(json.dumps(value,ensure_ascii=False).encode()).hexdigest()


def find_span(sequence,pattern):
    if not pattern:
        raise ValueError("Empty span marker")
    hits=[i for i in range(len(sequence)-len(pattern)+1)
          if sequence[i:i+len(pattern)]==pattern]
    if len(hits)!=1:
        raise ValueError(f"Expected one unambiguous marker, found {len(hits)}")
    return hits[0],hits[0]+len(pattern)


def evenly_sample(states,positions,n):
    if n<1 or len(positions)<1:
        raise ValueError("Token group empty")
    # Deterministic downsampling; repeats only when group has <n tokens.
    index=torch.linspace(0,len(positions)-1,n).round().long()
    return states[torch.as_tensor(positions,device=states.device)[index.to(states.device)]]


def extract_structured(states,input_ids,image_token_id,tokenizer,counts):
    """Source: image placeholder IDs, definition span, relation span.

    The sequence is the multimodal processor's expanded input_ids, not
    tokenization of text alone; image positions therefore reflect the
    actual number of MLLM visual-language tokens.
    """
    if states.ndim!=2 or len(input_ids)!=len(states):
        raise ValueError("Hidden states and token IDs length mismatch")
    ids=[int(x) for x in input_ids]
    image_positions=[i for i,t in enumerate(ids) if t==int(image_token_id)]
    marks=[tokenizer.encode(marker,add_special_tokens=False)
           for marker in (START_DEF,START_REL,END_REL)]
    ranges=[find_span(ids,m) for m in marks]
    if not (ranges[0][1]<ranges[1][0]<ranges[1][1]<ranges[2][0]):
        raise ValueError("MLLM prompt segment order is invalid")
    definition_positions=list(range(ranges[0][1],ranges[1][0]))
    relation_positions=list(range(ranges[1][1],ranges[2][0]))
    groups=[image_positions,definition_positions,relation_positions]
    if not all(groups):
        raise ValueError("Missing image/definition/relation token group")
    if len(counts)!=3 or any(n<1 for n in counts):
        raise ValueError("Three positive token budgets required")
    return torch.cat([evenly_sample(states,pos,n)
                      for pos,n in zip(groups,counts)],dim=0).detach().float().cpu()


class StructuredFrozenMLLM(FrozenMLLM):
    @torch.inference_mode()
    def encode(self,image_path,task,counts=(32,16,16)):
        with Image.open(image_path) as img:
            message=[{"role":"user","content":[
                {"type":"image","image":img.convert("RGB")},
                {"type":"text","text":structured_prompt(task)}]}]
            inputs=self.processor.apply_chat_template(
                message,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors="pt").to(self.device)
        states=self.model(**inputs,output_hidden_states=True,
                          use_cache=False,return_dict=True).hidden_states[-1][0].float()
        image_id=self.model.config.image_token_id
        tokens=extract_structured(states,inputs["input_ids"][0].tolist(),
                                  image_id,self.processor.tokenizer,counts)
        if not torch.isfinite(tokens).all():
            raise ValueError("Nonfinite structured MLLM hidden states")
        return tokens
