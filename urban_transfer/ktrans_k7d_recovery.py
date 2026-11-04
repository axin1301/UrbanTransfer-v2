"""Auditable, source-only projection of already-generated Image-Only v2 JSON.

A v2 response occasionally contains unsolicited task relevance/proxy
reasoning despite its task-free prompt. Only accept fully valid, complete
visual_cues and evidence_confidence. Preserve the raw response and record
every discarded extra key. No task relevance is used downstream.
Malformed/truncated evidence is never fabricated or imputed.
"""
import json,re
from scripts.run_ktrans_k7_vlm_pilot import CUES
from urban_transfer.ktrans_k7d_image_only import parse_image_only

REQUIRED=('scene','visual_cues','evidence_confidence','limitations')
EXTRA_ALLOWED=('evidence_relevance','task_relevance','proxy_reasoning')

def project_image_only_v2(raw):
    strict=parse_image_only(raw)
    if strict is not None:
        return strict,(), 'strict'
    try:
        obj=json.loads(raw)
    except json.JSONDecodeError:
        found=re.search(r'\{.*\}',raw,re.S)
        if found:
            try:obj=json.loads(found.group(0))
            except json.JSONDecodeError:return None,(), 'invalid_json'
        else:return None,(), 'invalid_json'
    if not isinstance(obj,dict):return None,(), 'invalid_object'
    extras=tuple(sorted(set(obj)-set(REQUIRED)))
    if not extras or not set(extras).issubset(EXTRA_ALLOWED):
        return None,extras,'unexpected_schema'
    # Requirement: no external claims carried forward, even indirectly.
    # Drop ONLY enumerated irrelevant keys in a fresh isolated object.
    filtered={k:obj[k] for k in REQUIRED if k in obj}
    parsed=parse_image_only(json.dumps(filtered,ensure_ascii=False))
    if parsed is None:
        return None,extras,'invalid_visual_scores'
    return parsed,extras,'projected_extra_fields'
