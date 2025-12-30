import torch
from torch.utils.data import Dataset, DataLoader
from valuation_generator import ValuationGenerator
from allocation_algorithm import AllocationAlgorithm
from tqdm import tqdm

class SyntheticDataset(Dataset):
    def __init__(self,
                 num_agents:int,
                 num_objects:int,
                 num_samples:int,
                 generator:ValuationGenerator=None,
                 allocator:AllocationAlgorithm=None,
                 saved_path:str=None) -> None:
        super().__init__()
        assert num_agents > 0, f"number of agents must be positive, got: {num_agents}"
        assert num_objects > 0, f"number of agents must be positive, got: {num_objects}"
        
        if saved_path is None:
            self.valuation = generator.generate(size=(num_samples, num_agents, num_objects))
            print("Generate allocations")
            self.allocation = [allocator.allocate(valuation=v) for v in tqdm(self.valuation)]
        else:
            data = torch.load(saved_path)
            self.valuation = data['valuation']
            self.allocation = data['allocation']
            assert self.valuation[0].shape == self.allocation[0].shape == (num_agents, num_objects), f"Shape not equal: {self.valuation[0].shape}, {self.allocation[0].shape}, {(num_agents, num_objects)}"
            self.valuation = self.valuation[:num_samples]
            self.allocation = self.allocation[:num_samples]

    def __len__(self):
        return len(self.valuation)

    def __getitem__(self, idx):
        valuation = self.valuation[idx]
        allocation = self.allocation[idx]

        return valuation, allocation
    
    def save(self, save_to:str):
        torch.save({
            'valuation': self.valuation,
            'allocation': torch.stack(self.allocation) 
        }, save_to)

def create_loader(dataset:Dataset, *args, **kwargs):
    return DataLoader(dataset=dataset, *args, **kwargs)
