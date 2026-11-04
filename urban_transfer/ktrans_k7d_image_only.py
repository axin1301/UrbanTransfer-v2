"""K7-D Image-Only v2 schema: no task relevance field by design.

Only observable scene morphology and evidence confidence are requested.
No hidden task prior can contaminate the image-only ablation. This protocol
is separately keyed; image+task and text-only K7-B caches remain reusable.
"""
import json,re
from scripts.run_ktrans_k7_vlm_pilot import CUES

IMAGE_ONLY_VERSION='k7d-image-only-v2'
IMAGE_ONLY_SYSTEM=(
    'You are a satellite scene observer, NOT an indicator or transfer analyst. '
    'No task has been specified. Only describe visible overhead urban morphology. '
    'Return only a JSON object using the required fields; never invent indicators, '
    'schools, cellular infrastructure, measured urban values or relevance scores.'
)
IMAGE_ONLY_PROMPT=(
    'IMAGE-ONLY OBSERVATION PROTOCOL k7d-image-only-v2. There is NO target task. '
    'Return ONLY a JSON object with EXACT keys: scene, visual_cues, '
    'evidence_confidence, limitations. '
    'Both visual_cues and evidence_confidence MUST be objects with these exact six '
    'keys: '+', '.join(CUES)+'. '
    'Each score is 0=not visible/no support, 1=weak, 2=moderate, 3=strong, '
    'or null=not visually assessable. '
    'Do not include evidence_relevance, task_relevance, proxy_reasoning '
    'or any task-specific scoring field. '
    'Describe only observable appearance, not function inferred from labels. '
    'scene must be at most 40 words; limitations must be a short sentence.'
)

def parse_image_only(raw):
    try:
        obj=json.loads(raw)
    except json.JSONDecodeError:
        found=re.search(r'\{.*\}',raw,re.S)
        if not found:return None
        try:obj=json.loads(found.group(0))
        except json.JSONDecodeError:return None
    if not isinstance(obj,dict):return None
    if any(key in obj for key in ('evidence_relevance','task_relevance','proxy_reasoning')):
        return None  # fail closed: must not turn generated relevance into observations
    result={}
    for field in ('visual_cues','evidence_confidence'):
        values=obj.get(field)
        if field=='evidence_confidence' and field in obj and values is None:
            # An entire null confidence dictionary means every cue is
            # unassessable, NOT zero confidence and NOT a fabricated score.
            values={cue:None for cue in CUES}
        if not isinstance(values,dict) or set(values)!=set(CUES):
            return None
        parsed={}
        for cue in CUES:
            value=values[cue]
            if value is not None and (type(value) is not int or not 0<=value<=3):
                return None
            parsed[cue]=value
        result[field]=parsed
    if not isinstance(obj.get('scene'),str) or not isinstance(obj.get('limitations'),str):
        return None
    result['scene']=obj['scene'][:400]
    result['limitations']=obj['limitations'][:400]
    return result
