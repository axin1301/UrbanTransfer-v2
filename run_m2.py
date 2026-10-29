"""M2.1 train and evaluation, with independent checkpoint/evaluation commands."""
import argparse
from urban_transfer.m2 import train,evaluate
p=argparse.ArgumentParser()
p.add_argument('command',choices=['train','evaluate'])
p.add_argument('--train-csv')
p.add_argument('--scui-csv');p.add_argument('--ucsi-csv');p.add_argument('--ucui-csv')
p.add_argument('--image-root')
p.add_argument('--image-column')
p.add_argument('--provider',choices=['no-llm','qwen3vl-latent','shuffled-latent','qwen3vl','mock'],default='qwen3vl-latent')
p.add_argument('--model-id',default='Qwen/Qwen3-VL-2B-Instruct')
p.add_argument('--output',default='runs/m2');p.add_argument('--cache-dir',default='runs/qwen_cache')
p.add_argument('--no-cache',action='store_true')
p.add_argument('--checkpoint');p.add_argument('--resume')
p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=2)
p.add_argument('--limit',type=int);p.add_argument('--seed',type=int,default=42)
p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--val-fraction',type=float,default=0.1)
p.add_argument('--task-dim',type=int,default=128);p.add_argument('--visual-dim',type=int,default=768)
p.add_argument('--hidden-dim',type=int,default=128);p.add_argument('--experts',type=int,default=6)
p.add_argument('--top-k',type=int,default=2)
p.add_argument('--allow-missing-images',action='store_true',help='Mock smoke tests only')
args=p.parse_args()
if args.command=='train':
    if not args.train_csv:p.error('--train-csv is required for training')
    train(args)
else:
    evaluate(args)
