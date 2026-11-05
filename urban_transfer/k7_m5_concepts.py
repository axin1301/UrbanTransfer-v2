"""One-off cached concept text anchors in BOTH frozen encoder spaces.

RemoteCLIP text vectors align with existing cached RemoteCLIP IMAGE vectors.
Qwen concept vectors align with existing Qwen directed relation vectors.
Never mix the two embedding spaces directly.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

CONCEPTS=(
    ("residential", "residential housing distribution and neighborhoods",
     "a satellite image of dense residential houses and apartment buildings"),
    ("settlement", "human settlement morphology and urban built up density",
     "a satellite image of dense urban settlement and built-up blocks"),
    ("roads", "roads, intersections and transportation network connectivity",
     "a satellite image of roads, intersections and street networks"),
    ("commercial", "commercial buildings and local service facility structures",
     "a satellite image of commercial buildings, shops and service facilities"),
)


def cache_path(root,qwen_model,remoteclip_model):
    payload=json.dumps({
        "concepts":CONCEPTS,"qwen":qwen_model,"remoteclip":remoteclip_model,
        "version":1,"qwen_pool":"last-token","clip_pool":"encode_text",
    },sort_keys=True)
    digest=hashlib.sha256(payload.encode()).hexdigest()[:20]
    return Path(root)/f"k7_m5_concept_anchors_{digest}.pt"


def get_concept_anchors(root,qwen_model,remoteclip_model,visual_dim,semantic_dim):
    path=cache_path(root,qwen_model,remoteclip_model)
    if path.is_file():
        cached=torch.load(path,map_location="cpu",weights_only=True)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU needed to encode first-time M5 concepts")
        from transformers import AutoTokenizer,AutoModel
        from urban_transfer.ktrans_text import last_token_pool
        print("[k7-m5] encoding 4 concepts in Qwen space...",flush=True)
        tokenizer=AutoTokenizer.from_pretrained(qwen_model,padding_side="left")
        model=AutoModel.from_pretrained(
            qwen_model,torch_dtype=torch.bfloat16).to("cuda").eval()
        texts=[concept[1] for concept in CONCEPTS]
        with torch.inference_mode():
            inputs=tokenizer(texts,padding=True,truncation=True,
                max_length=256,return_tensors="pt").to("cuda")
            hidden=model(**inputs).last_hidden_state
            qwen=F.normalize(
                last_token_pool(hidden,inputs["attention_mask"]).float(),
                dim=-1).cpu()
        del model
        torch.cuda.empty_cache()

        # Reuse exact RemoteCLIP model and checkpoint family that encoded
        # the existing tile cache. No image re-encoding is performed.
        print("[k7-m5] encoding 4 concepts in RemoteCLIP text space...",flush=True)
        from urban_transfer.m3_features import FrozenRemoteCLIP
        import open_clip
        encoder=FrozenRemoteCLIP(model_name=remoteclip_model,device="cuda")
        tokenizer_clip=open_clip.get_tokenizer(remoteclip_model)
        with torch.inference_mode():
            clip_tokens=tokenizer_clip([concept[2] for concept in CONCEPTS]).to("cuda")
            clip=F.normalize(encoder.model.encode_text(clip_tokens).float(),
                             dim=-1).cpu()
        del encoder
        torch.cuda.empty_cache()
        cached={"qwen":qwen,"remoteclip":clip}
        path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(cached,path)
        print(f"[k7-m5] concept anchors cached {path}",flush=True)

    if (set(cached)!={"qwen","remoteclip"} or
        cached["qwen"].shape!=(4,semantic_dim) or
        cached["remoteclip"].shape!=(4,visual_dim) or
        not all(torch.isfinite(t).all() for t in cached.values())):
        raise ValueError("M5 text concept cache shape/dimension mismatch")
    return F.normalize(cached["qwen"].float(),dim=-1),F.normalize(
        cached["remoteclip"].float(),dim=-1)


def visual_concept_targets(train_visual,all_visual,clip_anchors):
    """Pseudo-labels from frozen aligned RemoteCLIP, calibrated SOURCE-only.

    Input all_visual may include dev rows but means/stds come solely from
    train_visual to avoid heldout-feature statistics leakage.
    """
    src=F.normalize(train_visual.float(),dim=-1) @ clip_anchors.float().T
    all_scores=F.normalize(all_visual.float(),dim=-1) @ clip_anchors.float().T
    center=src.mean(0)
    scale=src.std(0,unbiased=False).clamp_min(.005)
    return torch.sigmoid((all_scores-center)/scale).detach()
