"""CPU smoke test with synthetic embeddings and synthetic knowledge (NOT a scientific experiment)."""
import torch
from urban_transfer import TransferModel,ModelConfig,MockProvider
torch.manual_seed(42)
cfg=ModelConfig(visual_dim=16,task_dim=8,hidden_dim=32,n_experts=4,top_k=2)
model=TransferModel(cfg);provider=MockProvider()
knowledge=provider.infer(image_path=None,city='test',task_description='test',source_tasks=('a','b','c','d','e'))
k=torch.tensor([knowledge.relevance+knowledge.morphology+[knowledge.uncertainty]]*4)
y,aux=model(torch.randn(4,16),torch.randn(4,4),torch.randn(4,8),k,True)
loss=y.square().mean();loss.backward()
print('prediction shape:',tuple(y.shape),'loss:',round(loss.item(),5))
print('selected expert IDs:',aux['expert_indices'].tolist())
