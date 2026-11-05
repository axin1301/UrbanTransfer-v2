"""Frozen Qwen2.5-VL per-(tile,indicator prompt) hidden-state token cache.

No language generation, no MLLM finetuning, no label in prompts. We store
last N prompt-conditioned language hidden states as fp16, not JSON answers.
SQLite resumes at individual image/task pairs and never overwrites entries.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from urban_transfer.k7_m1_knowledge import TRAIN_TASKS, HELDOUT_TASK, relation_text
from urban_transfer.task_profiles import PROFILES

TASKS=(*TRAIN_TASKS,HELDOUT_TASK)
VERSION="mllm-hidden-v1"


def task_prompt(task):
    if task not in TASKS:
        raise ValueError("Unknown MLLM pilot task: "+str(task))
    definition,measurement,_=PROFILES[task]
    info=[
        "Interpret the satellite tile ONLY as observable spatial evidence.",
        "Target urban indicator: "+task,
        "Indicator definition: "+definition,
        "Measurement context: "+measurement,
        "Which visible spatial patterns could help this indicator's regression?",
        "Use source-to-target transfer hypotheses cautiously. Do not guess numerical indicator values.",
    ]
    for source in TRAIN_TASKS:
        if source!=task:
            info.append(relation_text(source,task))
    info.append("Internal representation requested: spatial visual clues relevant to the defined indicator.")
    return "\n".join(info)


def feature_key(path,task,model_id,tokens,max_pixels):
    p=Path(path).resolve()
    stat=p.stat()
    config=json.dumps([VERSION,str(p),stat.st_size,stat.st_mtime_ns,task,
                       task_prompt(task),model_id,tokens,max_pixels],
                      ensure_ascii=False)
    return hashlib.sha256(config.encode()).hexdigest()


class HiddenStore:
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.path,timeout=60) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS hidden (
                 cache_key TEXT PRIMARY KEY, task TEXT NOT NULL,
                 n_tokens INTEGER NOT NULL, dim INTEGER NOT NULL,
                 values_blob BLOB NOT NULL)""")

    def get(self,key):
        with sqlite3.connect(self.path,timeout=60) as conn:
            row=conn.execute(
                "SELECT n_tokens,dim,values_blob FROM hidden WHERE cache_key=?",
                (key,)).fetchone()
        if row is None:return None
        n,d,blob=row
        if len(blob)!=n*d*2:
            raise ValueError("Corrupt MLLM cache blob")
        return torch.from_numpy(np.frombuffer(blob,dtype="<f2")
                                .astype("float32").reshape(n,d).copy())

    def put(self,key,task,hidden):
        tensor=hidden.detach().float().cpu()
        if tensor.ndim!=2 or not torch.isfinite(tensor).all():
            raise ValueError("MLLM cache expects finite [tokens,hidden] tensor")
        arr=tensor.numpy().astype("<f2")
        with sqlite3.connect(self.path,timeout=60) as conn:
            conn.execute("INSERT OR IGNORE INTO hidden VALUES (?,?,?,?,?)",
                         (key,task,arr.shape[0],arr.shape[1],
                          sqlite3.Binary(arr.tobytes())))


class FrozenMLLM:
    def __init__(self,model_id="Qwen/Qwen2.5-VL-3B-Instruct",
                 max_pixels=128*28*28,device="cuda"):
        if not torch.cuda.is_available() and device=="cuda":
            raise RuntimeError("MLLM encoding needs MI250 ROCm GPU")
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.device=torch.device(device)
        self.processor=AutoProcessor.from_pretrained(
            model_id,min_pixels=56*56,max_pixels=max_pixels)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,torch_dtype=torch.bfloat16,
            attn_implementation="eager").to(self.device).eval()

    @torch.inference_mode()
    def encode(self,image_path,task,n_tokens=8):
        # Use the official Qwen multimodal chat template; no generated text.
        with Image.open(image_path) as img:
            image=img.convert("RGB")
            dialogue=[{"role":"user","content":[
                {"type":"image","image":image},
                {"type":"text","text":task_prompt(task)}]}]
            encoded=self.processor.apply_chat_template(
                dialogue,tokenize=True,add_generation_prompt=True,
                return_dict=True,return_tensors="pt").to(self.device)
        outputs=self.model(**encoded,output_hidden_states=True,
                           use_cache=False,return_dict=True)
        states=outputs.hidden_states[-1][0].float()
        mask=encoded["attention_mask"][0].bool()
        selected=states[mask]
        if selected.shape[0]<n_tokens:
            raise ValueError("Prompt has fewer valid tokens than requested")
        # Last prompt positions are contextualized by all image and text tokens.
        pooled=selected[-n_tokens:].detach().cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError("MLLM forward produced nonfinite hidden states")
        return pooled
