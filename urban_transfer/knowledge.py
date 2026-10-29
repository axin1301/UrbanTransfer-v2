"""Online, input-conditioned knowledge acquisition. No target labels enter prompts."""
from dataclasses import dataclass
from typing import Protocol
import json
import re

TASKS = ('population_count','nighttime_light_intensity','education_accessibility','food_accessibility','digital_total_cell_intensity', 'digital_modernization_tmi')

@dataclass
class Knowledge:
    relevance: list[float]
    morphology: list[float]
    uncertainty: float
    explanation: str = ''

    def validate(self, n_tasks: int):
        if len(self.relevance) != n_tasks or len(self.morphology) != 3:
            raise ValueError('Knowledge dimensions mismatch')
        import math
        if not all(math.isfinite(float(x)) and 0 <= float(x) <= 1 for x in self.relevance+self.morphology+[self.uncertainty]):
            raise ValueError('Knowledge scores must be finite in [0,1]')
        return self

class KnowledgeProvider(Protocol):
    def infer(self, *, image_path: str | None, city: str, task_description: str, source_tasks: tuple[str,...]) -> Knowledge: ...

class MockProvider:
    """ONLY for pipeline tests; contains no LLM knowledge."""
    def infer(self, *, image_path, city, task_description, source_tasks):
        return Knowledge([0.5]*len(source_tasks), [0.5]*3, 0.5, 'Synthetic test knowledge')

class TransformersVisionProvider:
    """Calls a local HuggingFace multimodal model at EACH infer invocation."""
    def __init__(self, model_id: str, device: str = 'cpu', max_new_tokens: int = 160):
        from transformers import pipeline
        self.pipe = pipeline('image-text-to-text', model=model_id, device=device)
        self.max_new_tokens = max_new_tokens

    def infer(self, *, image_path, city, task_description, source_tasks):
        if image_path is None:
            raise ValueError('Online MLLM requires a real image_path; cached embeddings alone are insufficient.')
        from PIL import Image
        with Image.open(image_path) as img:
            image = img.convert('RGB')
            prompt = (
              'You are given a satellite image and a target indicator definition. '
              'Identify potentially useful source indicators and visually grounded morphology. '
              'Do NOT invent observed facilities, counts, labels or numeric target values. '
              'Scores are tentative transfer priors, not ground truth. '
              f'City: {city}. Target definition: {task_description}. '
              f'Source indicator names in fixed order: {list(source_tasks)}. '
              'Return ONLY JSON with keys relevance (array of numbers 0..1 in source order), '
              'morphology (3 numbers 0..1 for urban density, visible roads, mixed land use), '
              'uncertainty (0..1), explanation (short string).'
            )
            result = self.pipe(images=image, text=prompt, max_new_tokens=self.max_new_tokens)
        raw = result[0].get('generated_text','')
        if isinstance(raw,list):
            raw = ' '.join(str(m.get('content','')) for m in raw if isinstance(m,dict))
        match = re.search(r'\{.*\}',str(raw),re.S)
        if not match:
            raise ValueError(f'MLLM output is not JSON: {str(raw)[:250]}')
        obj = json.loads(match.group(0))
        return Knowledge(**{k:obj[k] for k in ('relevance','morphology','uncertainty','explanation')}).validate(len(source_tasks))
