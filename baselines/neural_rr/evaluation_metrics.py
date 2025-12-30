from abc import ABCMeta, abstractmethod
import torch
from torch import Tensor
import torch.nn as nn
from typing import List

class AllocationMetric(nn.Module, metaclass=ABCMeta):
    def __init__(self, normalize:bool = True):
        super().__init__()
        self.normalize = normalize

    def _assert_dimension(self, output: Tensor):
        assert output.ndim == 2, f"output are assumed to be 1 instance, got: {output.ndim=}"

    def _assert_shape_equal(self, output: Tensor, label: Tensor):
        assert output.ndim == label.ndim, f"output and label must have the same dimensions, got: {output.ndim=}, {label.ndim=}"
        assert output.shape == label.shape, f"output and label must have the same shape, got: {output.shape=}, {label.shape=}"

    def _assert_integrality(self, output: Tensor):
        assert output.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
        assert torch.all((output == 0) | (output == 1)), "Output contains elements other than 0 and 1"
        assert torch.all(output.sum(dim=0) <= 1), f"One item must be allocated up to one agent, got:{output.sum(dim=0)=}"

    @abstractmethod
    def _forward(self, valuation: Tensor, output: Tensor, label: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, valuations: Tensor, outputs: Tensor, labels: Tensor) -> List[float]:
        return [self._forward(valuation, output, label).item() for (valuation, output, label) in zip(valuations, outputs, labels)]

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

class HammingDistance(AllocationMetric):
    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _forward(self, valuation: Tensor, output: Tensor, label: Tensor) -> Tensor:
        self._assert_dimension(output=output)
        self._assert_shape_equal(output=output, label=label)
        self._assert_integrality(output=output)

        _, m = output.shape
        metric = (output - label).abs().sum() / (2*m * self.normalize + 1.0 * (1-self.normalize))

        return metric

    def __repr__(self):
        name = 'Hamm'
        if self.normalize:
            name += '(normalize)'
        return name

class UtilitarianWelfare(AllocationMetric):
    def __init__(self, normalize: bool=True, relative: bool=True):
        super().__init__(normalize=normalize)
        self.relative = relative

    def _forward(self, valuation: Tensor, output: Tensor, label: Tensor) -> Tensor:
        self._assert_dimension(output=output)
        self._assert_integrality(output=output)

        n, m = valuation.shape
        norm_factor = (n * m * self.normalize + 1.0 * (1-self.normalize))
        metric = (valuation * output).sum() / norm_factor

        if self.relative:
            gt_metric = (valuation * label).sum() / norm_factor
            metric = metric / gt_metric

        return metric

    def __repr__(self):
        name = 'SCW'
        if self.normalize:
            if self.relative:
                name += '(normalize, relative)'
            else:
                name += '(normalize)'
        else:
            if self.relative:
                name += '(relative)'

        return name

class EF1Hard(AllocationMetric):
    def __init__(self, normalize: bool = True):
        super().__init__(normalize=normalize)

    def _forward(self, valuation: Tensor, output: Tensor, label: Tensor) -> Tensor:
        self._assert_dimension(output=output)
        self._assert_integrality(output=output)

        n, _ = output.shape
        for i in range(n):
            for j in range(n):
                if i == j: continue
                envy = (valuation[i] * output[j]).sum().item() - (valuation[i] * output[i]).sum().item()
                is_envious = (envy > 0)
                if is_envious:
                    for o in torch.where(output[j])[0].tolist():
                        removed_output_j = self.__safe_replace(tensor=output[j], target_index=o)
                        envy = (valuation[i] * removed_output_j).sum().item() - (valuation[i] * output[i]).sum().item()
                        is_envious = is_envious and (envy > 0)
                    if is_envious:
                        return torch.tensor(0.0)
        return torch.tensor(1.0)

    def __safe_replace(self, tensor: Tensor, target_index, target_value=1, new_value=0) -> Tensor:
        assert tensor.ndim == 1

        if tensor[target_index] == target_value:
            new_tensor = torch.cat([tensor[:target_index], torch.tensor([new_value]), tensor[target_index+1:]])

            return new_tensor
        else:
            return tensor

    def __repr__(self):
        return "EF1"
