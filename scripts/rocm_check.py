import torch
print('torch:',torch.__version__,'HIP:',torch.version.hip,'CUDA namespace available:',torch.cuda.is_available())
if not torch.cuda.is_available() or not torch.version.hip: raise SystemExit('ROCm PyTorch is required for MI250X')
print('devices:',torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i,torch.cuda.get_device_name(i))
x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16);print('bf16 matmul:',(x@x).mean().item())
