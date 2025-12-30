import torch
import torch.nn as nn
import numpy as np
import random
import os
from allocation_algorithm import MaximumUtilitarianWelfare
from evaluation_metrics import HammingDistance, UtilitarianWelfare, EF1Hard
from loss import ItemCrossEntropy, EFViolation, Regularized
from valuation_generator import AverageNoise
from typing import Literal

def select_allocator(choice:Literal['MUW'], **kwargs):
    if choice == 'MUW':
        return MaximumUtilitarianWelfare()
    else:
        assert False

def select_loss_functions(choice:Literal['ItemCE', 'EFViolation', 'Reg'], **kwargs):
    if choice == 'ItemCE':
        return ItemCrossEntropy(kwargs.get('reduction', 'mean'))
    elif choice == 'EFViolation':
        return EFViolation()
    elif choice == 'Reg':
        assert 'lams' in kwargs and 'losses' in kwargs
        lams = kwargs['lams']
        losses = [select_loss_functions(choice=key, **val) for (key, val) in kwargs['losses'].items()]
        return Regularized(lams, losses=losses)
    else:
        assert False

def select_valuations(choice:Literal['AverageNoise'], **kwargs):
    if choice == 'AverageNoise':
        return AverageNoise(low=kwargs.get('low', 1.0), high=kwargs.get('high', 2.0), eps=kwargs.get('eps', 1e-2))
    else:
        assert False

def select_metrics(choice:Literal['Hamm', 'SCW', 'rSCW', 'EF1Hard'], normalize:bool=True):
    if choice == 'Hamm':
        return HammingDistance(normalize=normalize)
    elif choice == 'SCW':
        return UtilitarianWelfare(normalize=normalize, relative=False)
    elif choice == 'rSCW':
        return UtilitarianWelfare(normalize=normalize, relative=True)
    elif choice == 'EF1Hard':
        return EF1Hard(normalize=normalize)
    else:
        assert False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed:int=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed)

    return g

def k_hot_encode(lists, num_cols):
    """
    Converts a list of index lists into a k-hot encoded tensor.

    Args:
        lists (list of list of int): A list where each element is a list of column indices to be set to 1.
        num_cols (int): The number of columns in the resulting tensor.

    Returns:
        torch.Tensor: A 2D tensor of shape (len(lists), num_cols) with k-hot encoding.

    Example:
        >>> k_hot_encode([[1], [], [2, 3], [0]], 4)
        tensor([[0., 1., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 1., 1.],
                [1., 0., 0., 0.]])
    """
    num_rows = len(lists)
    E = torch.zeros((num_rows, num_cols), dtype=torch.float32)
    for i, sublist in enumerate(lists):
        E[i, sublist] = 1.0
    return E

def save_model(model:nn.Module, directory:str, filename:str):
    os.makedirs(directory, exist_ok=True)

    file_path = os.path.join(directory, filename)

    # save model
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
