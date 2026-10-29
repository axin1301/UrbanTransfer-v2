"""Read-only checks: reuse pre-existing ROCm environment when possible."""
from importlib.metadata import version, PackageNotFoundError
import sys
names=['torch','torchvision','transformers','accelerate','numpy','pandas','Pillow','pyarrow']
for name in names:
    try: print(f'{name}: {version(name)}')
    except PackageNotFoundError: print(f'{name}: NOT INSTALLED')
try:
    import torch
    print('torch.cuda.is_available:',torch.cuda.is_available())
    if torch.cuda.is_available(): print('GPU:',torch.cuda.get_device_name(0))
    print('torch.version.hip:',torch.version.hip)
except Exception as e: print('Torch import failed:',repr(e))
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    print('Qwen3-VL classes: import OK')
except Exception as e:
    print('Qwen3-VL classes: import failed:',repr(e))
    sys.exit(1)
