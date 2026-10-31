"""Versioned, frozen Qwen3 embedding cache. Text only; no image inference."""
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from .task_profiles import task_texts

def cache_path(root,model_id,profile,texts):
    payload=json.dumps({'model':model_id,'profile':profile,'texts':texts,'pool':'last-token','v':1},sort_keys=True,ensure_ascii=False)
    return Path(root)/('qwen3_'+hashlib.sha256(payload.encode()).hexdigest()[:20]+'.pt')

def last_token_pool(hidden,mask):
    if bool((mask[:,-1]==1).all()):
        return hidden[:,-1,:]
    idx=mask.sum(dim=1)-1
    return hidden[torch.arange(hidden.shape[0],device=hidden.device),idx]

def qwen_task_vectors(*,root,model_id='Qwen/Qwen3-Embedding-0.6B',
                      profile='short',llm_profile_file=None,only_cached=False,
                      max_length=2048,batch_size=4,device=None):
    texts=task_texts(profile,llm_profile_file)
    output=cache_path(root,model_id,profile,texts)
    if output.exists():
        vectors=torch.load(output,map_location='cpu',weights_only=True)
        if set(vectors)==set(texts) and all(torch.isfinite(x).all() for x in vectors.values()):
            return vectors
        raise ValueError('Invalid Qwen embedding cache: '+str(output))
    if only_cached:
        raise FileNotFoundError(f'Qwen task cache missing; precompute first: {output}')
    if max_length<1 or batch_size<1:raise ValueError('Invalid token limit or batch size')
    try:
        from transformers import AutoTokenizer,AutoModel
    except ImportError as exc:
        raise RuntimeError('Qwen task precompute requires transformers>=4.51.0') from exc
    target=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    tokenizer=AutoTokenizer.from_pretrained(model_id,padding_side='left')
    model=AutoModel.from_pretrained(model_id,torch_dtype=torch.bfloat16 if target.type=='cuda' else torch.float32)
    model=model.to(target).eval()
    names=sorted(texts)
    result={}
    with torch.inference_mode():
        for start in range(0,len(names),batch_size):
            batch_names=names[start:start+batch_size]
            inputs=tokenizer([texts[n] for n in batch_names],padding=True,truncation=True,
                             max_length=max_length,return_tensors='pt').to(target)
            output_hidden=model(**inputs).last_hidden_state
            embeddings=F.normalize(last_token_pool(output_hidden,inputs['attention_mask']).float(),p=2,dim=-1).cpu()
            for i,name in enumerate(batch_names):
                result[name]=embeddings[i].clone()
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,output)
    print(f'[k4] wrote {len(result)} Qwen3 text embeddings to {output}',flush=True)
    return result
