"""Frozen Qwen relation embeddings and fold-safe relation aggregation for Z1.

Relation cards are hypotheses, not labels. Never condition a training task on
the held-out pseudo-task's relation card; use only the current source registry.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from urban_transfer.protocol import SOURCE_TASKS
from urban_transfer.ktrans_p2e1_indicator_spec import catalog, relationship


def relation_texts():
    spec = catalog()
    texts = {}
    for target in sorted(SOURCE_TASKS):
        for source in sorted(SOURCE_TASKS):
            if target == source:
                continue
            edge = relationship(target, source, spec)
            texts[(target, source)] = (
                f"Urban target indicator: {target}. "
                f"Target definition: {spec[target]['definition']}. "
                f"Target measurement: {spec[target]['measurement_context']}. "
                f"Candidate related indicator: {source}. "
                f"Source definition: {spec[source]['definition']}. "
                f"Source measurement: {spec[source]['measurement_context']}. "
                f"Conceptual relation ({edge['type']}): {edge['hypothesis']} "
                f"Limitations: {edge['important_difference']} "
                "This is an unverified conceptual hypothesis, NOT a measured "
                "predictive correlation or expert competence. "
                "Raw units and exact formulas remain unverified."
            )
    return texts


def cache_file(root, model_id):
    payload = json.dumps({"model": model_id, "texts": {
        "|".join(k): v for k, v in relation_texts().items()},
        "pool": "last-token", "version": 1}, sort_keys=True)
    return Path(root) / ("z1_relation_" +
                         hashlib.sha256(payload.encode()).hexdigest()[:20] + ".pt")


def relation_embeddings(root, model_id="Qwen/Qwen3-Embedding-0.6B",
                        *, only_cached=True, device=None, batch_size=4):
    path = cache_file(root, model_id)
    texts = relation_texts()
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        result = {tuple(k.split("|")): v for k, v in saved.items()}
        if set(result) != set(texts) or not all(
                torch.isfinite(x).all() for x in result.values()):
            raise ValueError("Corrupt or mismatched Z1 relation cache")
        return {k: F.normalize(v.float().flatten(), dim=0)
                for k, v in result.items()}
    if only_cached:
        raise FileNotFoundError(f"Missing Z1 relation cache: {path}")
    from transformers import AutoModel, AutoTokenizer
    from urban_transfer.ktrans_text import last_token_pool
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if target.type == "cuda"
        else torch.float32).to(target).eval()
    keys = sorted(texts)
    result = {}
    with torch.inference_mode():
        for start in range(0, len(keys), batch_size):
            subset = keys[start:start + batch_size]
            batch = tokenizer([texts[k] for k in subset], padding=True,
                              truncation=True, max_length=1024,
                              return_tensors="pt").to(target)
            hidden = model(**batch).last_hidden_state
            vec = F.normalize(last_token_pool(hidden, batch["attention_mask"]).float(),
                              dim=-1).cpu()
            for key, v in zip(subset, vec):
                result[key] = v.clone()
            print(f"[z1-rel-cache] {min(start+batch_size,len(keys))}/{len(keys)}",
                  flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"|".join(k): v for k, v in result.items()}, path)
    print(f"[z1-rel-cache] saved {path}", flush=True)
    return result


def fold_relations(embeddings, held_task, all_tasks, seed=42, shuffle=False):
    """No held-out target in train nodes' relation contexts.

    Shuffle control permutes relation messages among source *identities*
    within each target row; source names and all target definitions remain
    unchanged. As relations may share identical descriptions, assess the
    strength of this negative control rather than assume it is decisive.
    """
    tasks = tuple(sorted(all_tasks))
    if held_task not in tasks:
        raise ValueError("Unknown heldout task")
    eligible = tuple(t for t in tasks if t != held_task)
    answer = {}
    for target in tasks:
        sources = tuple(t for t in eligible if t != target)
        if not sources:
            raise ValueError("No eligible relation sources")
        links = [embeddings[(target, source)] for source in sources]
        if shuffle:
            # Rotate messages between nodes; if all edges are identical the
            # permutation must remain uninformative (explicit limitation).
            links = links[1:] + links[:1]
        answer[target] = F.normalize(torch.stack(links).mean(0), dim=0)
    return answer
