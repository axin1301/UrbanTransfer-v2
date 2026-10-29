"""Actual local Qwen3-VL multimodal generation, per input, not a pretend offline embedding."""
import json, re
import torch
from pathlib import Path
from .knowledge import Knowledge, parse_knowledge_json

class Qwen3VLProvider:
    def __init__(self, model_id='Qwen/Qwen3-VL-2B-Instruct', dtype='bfloat16', max_new_tokens=256, max_pixels=262144):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        if self.device=='cpu': raise RuntimeError('Real Qwen3-VL requires GPU; no CPU silent fallback. Use --provider mock for tests.')
        kwargs={'torch_dtype':torch.bfloat16 if dtype=='bfloat16' else torch.float16,'attn_implementation':'sdpa'}
        try:
            self.model=Qwen3VLForConditionalGeneration.from_pretrained(model_id,dtype=kwargs['torch_dtype'],attn_implementation='sdpa').to(self.device).eval()
        except TypeError:
            self.model=Qwen3VLForConditionalGeneration.from_pretrained(model_id,**kwargs).to(self.device).eval()
        self.processor=AutoProcessor.from_pretrained(model_id)
        self.max_new_tokens=max_new_tokens
        self.max_pixels=max_pixels
    @torch.inference_mode()
    def infer(self, *, image_path, city, task_description, source_tasks):
        from PIL import Image
        if not image_path or not Path(image_path).is_file(): raise FileNotFoundError(f'Qwen requires real image: {image_path}')
        with Image.open(image_path) as im: image=im.convert('RGB')
        prompt=("Analyze the supplied satellite image as evidence, not as ground truth facility inventory. "
          "Estimate which SOURCE indicator's learned model representations may help predict the TARGET indicator. "
          "Do not assume hidden labels or invent hospitals, schools or counts. "
          f"City metadata: {city}. Target indicator definition: {task_description}. "
          f"Ordered SOURCE indicators: {list(source_tasks)}. "
          "Return ONLY compact JSON, no markdown. 'relevance' MUST be a JSON ARRAY with exactly "+str(len(source_tasks))+" numeric scores in [0,1], in source order; "
          "'morphology': three [0,1] values for built-up density, visible road structure, mixed land-use appearance; "
          "'uncertainty': one [0,1] score; 'explanation': max 6 words. No extra prose.")
        messages=[{'role':'user','content':[{'type':'image','image':image,'max_pixels':self.max_pixels},
                    {'type':'text','text':prompt}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                   return_dict=True,return_tensors='pt')
        inputs.pop('token_type_ids',None)
        inputs={k:v.to(self.device) if hasattr(v,'to') else v for k,v in inputs.items()}
        errors = []
        for attempt in range(2):
            tokens = self.max_new_tokens if attempt == 0 else max(384, self.max_new_tokens)
            generated = self.model.generate(**inputs, do_sample=False, max_new_tokens=tokens)
            decoded = self.processor.batch_decode(
                generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True
            )[0]
            try:
                return parse_knowledge_json(decoded, len(source_tasks))
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                errors.append(f"attempt {attempt+1}: {exc}; output={decoded[:300]!r}")
                if attempt == 0:
                    correction = ("Your output did not follow the schema. Return a single complete "
                        "minified JSON object, without markdown. relevance MUST be an array of "
                        + str(len(source_tasks)) + " numbers in [0,1]; morphology must be an array "
                        "of 3 numbers; uncertainty is one number; explanation max 5 words. "
                        "Do NOT output scalar relevance.")
                    retry_messages = messages + [
                        {'role':'assistant','content':[{'type':'text','text':decoded}]},
                        {'role':'user','content':[{'type':'text','text':correction}]}
                    ]
                    inputs = self.processor.apply_chat_template(
                        retry_messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors='pt'
                    )
                    inputs.pop('token_type_ids', None)
                    inputs = {k:v.to(self.device) if hasattr(v,'to') else v
                              for k,v in inputs.items()}
        raise ValueError("Qwen3-VL failed strict JSON validation after retry: " + " | ".join(errors))
