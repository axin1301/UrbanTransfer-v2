"""Frozen Qwen text-teacher soft targets for M4 label-free target KD.

These are *semantic similarity heuristics* over Qwen embeddings, not
empirically verified importance labels, causal claims or an LLM fine-tune.
"""
import torch
import torch.nn.functional as F


def teacher_source_distribution(source_text, target_text, directed_edges,
                                temperature=0.15, floor=0.1):
    """Teacher: two directed relations + source/target descriptions.

    Returns a detached two-component soft source allocation, never an
    Education regression pseudo-label. Smooth to avoid overconfident
    pseudo-supervision from uncalibrated cosine scores.
    """
    if (source_text.ndim!=2 or source_text.shape[0]!=2 or
        target_text.ndim!=1 or directed_edges.shape!=source_text.shape or
        target_text.numel()!=source_text.shape[1]):
        raise ValueError("Expected [2,D] sources, [D] target, [2,D] edges")
    if temperature<=0 or not 0<=floor<0.5:
        raise ValueError("Invalid teacher temperature/floor")
    src=F.normalize(source_text.float(),dim=-1)
    tgt=F.normalize(target_text.float(),dim=-1)
    relations=F.normalize(directed_edges.float(),dim=-1)
    # Directed mechanism agreement with target and source-task relatedness.
    scores=0.7*(relations@tgt)+0.3*(src@tgt)
    probs=F.softmax(scores/temperature,dim=0)
    probs=(1-2*floor)*probs+floor
    if not torch.isfinite(probs).all():
        raise ValueError("Nonfinite Qwen teacher distribution")
    return probs.detach()


def distillation_loss(student_weights, teacher_probs):
    """Categorical KL(teacher || student), averaged over source-only tiles."""
    if (student_weights.ndim!=2 or student_weights.shape[-1]!=2 or
        teacher_probs.shape!=(2,)):
        raise ValueError("Expected student [N,2] and teacher [2]")
    if (student_weights<=0).any() or not torch.isfinite(student_weights).all():
        raise ValueError("Invalid student probability distribution")
    target=teacher_probs.to(device=student_weights.device,
                            dtype=student_weights.dtype)
    return F.kl_div(student_weights.clamp_min(1e-8).log(),
                    target.unsqueeze(0).expand_as(student_weights),
                    reduction="batchmean")
