"""Frozen MLLM feature cache. Never cache labels, predictions, or trainable features."""
from pathlib import Path
import hashlib
import json
import os
import torch

class HiddenStateCache:
    def __init__(self, root, model_id, *, enabled=True):
        self.root=Path(root);self.model_id=model_id;self.enabled=enabled
        if enabled:self.root.mkdir(parents=True,exist_ok=True)

    def _key(self, *, image_path, city, task_description, source_tasks):
        image=Path(image_path).resolve()
        stat=image.stat()
        record={'schema':1,'model_id':self.model_id,
                'image_path':str(image),'image_bytes':stat.st_size,
                'image_mtime_ns':stat.st_mtime_ns,'city':city,
                'task_description':task_description,'source_tasks':list(source_tasks)}
        return hashlib.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def get_or_compute(self, *, provider, image_path, city, task_description, source_tasks):
        if not self.enabled:
            return provider.infer_hidden(image_path=image_path,city=city,
                                         task_description=task_description,source_tasks=source_tasks)
        key=self._key(image_path=image_path,city=city,task_description=task_description,source_tasks=source_tasks)
        path=self.root/(key+'.pt')
        if path.exists():
            try:
                value=torch.load(path,map_location='cpu',weights_only=True)
                if isinstance(value,torch.Tensor) and value.ndim==1 and torch.isfinite(value).all():
                    return value.float()
            except (RuntimeError,ValueError,EOFError,TypeError):
                pass
        value=provider.infer_hidden(image_path=image_path,city=city,
                                    task_description=task_description,source_tasks=source_tasks)
        if not isinstance(value,torch.Tensor) or value.ndim!=1 or not torch.isfinite(value).all():
            raise ValueError('MLLM latent vector must be finite 1D Tensor')
        value=value.detach().float().cpu()
        tmp=path.with_name(path.name+'.'+str(os.getpid())+'.tmp')
        try:
            torch.save(value,tmp)
            os.replace(tmp,path)
        except (OSError, RuntimeError) as exc:
            # Do not hide quota/permission/storage failures; otherwise a large
            # evaluation can silently rerun Qwen on every subsequent pass.
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(
                f"Failed to write Qwen cache entry at {path}: {exc}. "
                "Check df -h, df -i, filesystem quota and directory write permissions. "
                "Alternatively pass --cache-dir /path/with/space, or --no-cache "
                "(slower; Qwen reruns for every example)."
            ) from exc
        return value
