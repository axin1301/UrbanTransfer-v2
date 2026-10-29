"""Checkpoint format and deterministic RNG restore."""
from pathlib import Path
import random
import numpy as np
import torch

def save_checkpoint(path, *, model, optimizer, cfg, meta, epoch, best_val=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'config':vars(cfg),'meta':meta,'epoch':epoch,'best_val':best_val,
                'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),
                'python_rng':random.getstate(),
                'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)

def load_checkpoint(path, *, model, optimizer=None, restore_rng=False, device='cpu'):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    model.load_state_dict(ckpt['model'])
    if optimizer is not None:
        if 'optimizer' not in ckpt: raise ValueError('Checkpoint has no optimizer state; cannot resume training')
        optimizer.load_state_dict(ckpt['optimizer'])
    if restore_rng:
        if 'torch_rng' in ckpt: torch.set_rng_state(ckpt['torch_rng'].cpu())
        if 'numpy_rng' in ckpt: np.random.set_state(ckpt['numpy_rng'])
        if 'python_rng' in ckpt: random.setstate(ckpt['python_rng'])
        if ckpt.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    return ckpt
