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


def fold_relations(pair_embeddings, held_task, all_tasks, source_semantics,
                   shuffle=False):
    """Fold-safe source-identity × relation-message aggregation.

    For a held-out target, query relations to all training source indicators.
    For each training indicator, only refer to OTHER training indicators.
    Permuted control rotates relation cards among source identities while
    keeping target semantics and the source-identity vectors fixed. The
    identity-conditioned product makes this perturbation non-invariant.
    """
    tasks = tuple(sorted(all_tasks))
    if held_task not in tasks:
        raise ValueError("Unknown heldout task")
    eligible = tuple(t for t in tasks if t != held_task)
    if not set(tasks).issubset(source_semantics):
        raise ValueError("Missing source semantics for relation graph")
    answer = {}
    for target in tasks:
        sources = tuple(t for t in eligible if t != target)
        if len(sources) < 2:
            raise ValueError("Need multiple relation sources")
        messages = [F.normalize(pair_embeddings[(target, source)].float(), dim=0)
                    for source in sources]
        if shuffle:
            messages = messages[1:] + messages[:1]
        # Paired elementwise binding; NOT a simple mean of cards, for which
        # shuffling source relations would be mathematically invisible.
        bound = [F.normalize(source_semantics[source].float(), dim=0) * msg
                 for source, msg in zip(sources, messages)]
        answer[target] = F.normalize(torch.stack(bound).mean(0), dim=0)
    return answer

class RelationRegressor(torch.nn.Module):
    """Inject Qwen pairwise-relation messages into a shared zero-shot regressor.

    The relation message is fused BEFORE semantic FiLM / generated head;
    it is not a posthoc ensemble weight or supervised expert score.
    """
    def __init__(self, visual_dim, semantic_dim, mode="b3", hidden=128, rank=16):
        super().__init__()
        from urban_transfer.z1_model import Z1Regressor
        self.project = torch.nn.Sequential(
            torch.nn.LayerNorm(semantic_dim),
            torch.nn.Linear(semantic_dim, semantic_dim),
            torch.nn.Tanh())
        self.strength = torch.nn.Parameter(torch.tensor(0.20))
        self.predictor = Z1Regressor(visual_dim, semantic_dim, mode, hidden, rank)

    def forward(self, images, context, semantics, relations):
        if relations.shape != semantics.shape:
            raise ValueError("Relation vectors must match semantic embeddings")
        fused = F.normalize(
            semantics.float() + self.strength * self.project(relations.float()),
            dim=-1)
        return self.predictor(images, context, fused)
