"""Precompute the ten frozen Qwen3 task vectors, once per profile version."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from urban_transfer.ktrans_text import qwen_task_vectors

p=argparse.ArgumentParser()
p.add_argument('--text-model-id',default='Qwen/Qwen3-Embedding-0.6B')
p.add_argument('--task-profile',choices=['short','structured','llm'],default='structured')
p.add_argument('--llm-profile-file')
p.add_argument('--text-cache-dir',default='runs/ktrans_text_cache')
p.add_argument('--batch-size',type=int,default=4)
p.add_argument('--max-length',type=int,default=2048)
p.add_argument('--device')
args=p.parse_args()
vectors=qwen_task_vectors(root=args.text_cache_dir,model_id=args.text_model_id,
    profile=args.task_profile,llm_profile_file=args.llm_profile_file,
    batch_size=args.batch_size,max_length=args.max_length,device=args.device)
print(f'[k4] ready: {len(vectors)} tasks, dim={len(next(iter(vectors.values())))}')
