"""Actual local Qwen3-VL multimodal generation, per input, not a pretend offline embedding."""
import json, re
import torch
from pathlib import Path
from .knowledge import Knowledge

class Qwen3VLProvider:
    def __init__(self, model_id='Qwen/Qwen3-VL-2B-Instruct', dtype='bfloat16', max_new_tokens=160, max_pixels=262144):
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
          "Return only strict JSON with 'relevance': one score in [0,1] per source indicator in the supplied order; "
          "'morphology': three [0,1] values for built-up density, visible road structure, mixed land-use appearance; "
          "'uncertainty': one [0,1] score; 'explanation': a short string.")
        messages=[{'role':'user','content':[{'type':'image','image':image,'max_pixels':self.max_pixels},
                    {'type':'text','text':prompt}]}]
        inputs=self.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                   return_dict=True,return_tensors='pt')
        inputs.pop('token_type_ids',None)
        inputs={k:v.to(self.device) if hasattr(v,'to') else v for k,v in inputs.items()}
        generated=self.model.generate(**inputs,do_sample=False,max_new_tokens=self.max_new_tokens)
        decoded=self.processor.batch_decode(generated[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
        match=re.search(r'\{.*?\}',decoded,re.S)
        if not match: raise ValueError(f'No JSON in Qwen response: {decoded[:300]}')
        obj=json.loads(match.group(0))
        return Knowledge(relevance=obj['relevance'],morphology=obj['morphology'],uncertainty=obj['uncertainty'],
                         explanation=obj.get('explanation','')).validate(len(source_tasks))
