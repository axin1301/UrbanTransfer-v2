"""Deterministic dataset-wide permutation for Qwen knowledge intervention."""
import numpy as np
from torch.utils.data import Dataset

class PermutedKnowledgeDataset(Dataset):
    """Swap only MLLM input metadata, never the supervised image/target.

    A fixed derangement pairs every row with another row from the SAME partition.
    Inference and training remain deterministic with an explicit seed.
    """
    def __init__(self, dataset, seed=42):
        if len(dataset)<2:
            raise ValueError("Shuffled-latent intervention requires at least two rows")
        self.dataset=dataset
        n=len(dataset)
        permutation=np.random.default_rng(seed).permutation(n)
        # A random cycle guarantees no fixed points (unlike a plain permutation).
        shift=1+seed%(n-1)
        self.indices=np.roll(permutation,shift)
        # Use an explicit cycle in permuted order: each position maps elsewhere.
        self.indices=np.empty(n,dtype=np.int64)
        for i in range(n):
            self.indices[permutation[i]]=permutation[(i+1)%n]
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,index):
        sample=self.dataset[index]
        donor=self.dataset[int(self.indices[index])]
        sample['knowledge_image_path']=donor['image_path']
        sample['knowledge_city']=donor['city']
        sample['knowledge_task_description']=donor['task_description']
        return sample
