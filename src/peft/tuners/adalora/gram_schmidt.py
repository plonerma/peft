from typing import Generator, List, Literal, Tuple, Union, overload

import torch
from torch.nn import Parameter, ParameterList

from peft.tuners.lora import LoraLayer


@overload
def iter_vectors(
    params: Union[ParameterList, List[Parameter]],
    *,
    dim: int = 0,
    yield_indices: Literal[False] = False,
) -> Generator[torch.nn.Parameter, None, None]: ...


@overload
def iter_vectors(
    params: Union[ParameterList, List[Parameter]],
    *,
    dim: int = 0,
    yield_indices: Literal[True],
) -> Generator[Tuple[int, Tuple, torch.Tensor], None, None]: ...


def iter_vectors(
    params: Union[ParameterList, List[Parameter]],
    *,
    dim: int = 0,
    yield_indices: bool = False,
):
    for param_index, param in enumerate(params):
        for _index in range(param.size(dim)):
            index = [slice(None)] * param.dim()
            index[dim] = _index

            vector = param.select(dim, _index)
            if yield_indices:
                yield param_index, tuple(index), vector
            else:
                yield vector


def gram_schmidt_orthonormalize_parameters(
    params: Union[torch.nn.Parameter, torch.nn.Parameter], *, normalize: bool = True, dim: int = 0
):
    """Run Gram-Schmidt orthogonalization process and normalize."""

    if not isinstance(params, (torch.nn.ParameterList, list)):
        params = [params]

    for i, (param_index, vector_index, vector) in enumerate(
        iter_vectors(params, dim=dim, yield_indices=True)
    ):
        #offset = 0

        v = vector

        for j, vj in enumerate(iter_vectors(params, dim=dim)):
            if j >= i:
                break

            proj = torch.dot(v, vj) * vj

            if not normalize:
                proj /= torch.dot(vj, vj)

            v -= proj

        #v -= offset

        if normalize:
            length = torch.dot(v, v).sqrt()

            if length > 0.0:
                v /= length
        params[param_index][vector_index] = v


def gram_schmidt_orthonormalize_model(model, *, normalize: bool = True):
    for _, layer in model.named_modules():
        if isinstance(layer, LoraLayer):
            with torch.no_grad():
                for p in layer.lora_A.values():
                    gram_schmidt_orthonormalize_parameters(p, normalize=normalize, dim=0)

                for p in layer.lora_B.values():
                    gram_schmidt_orthonormalize_parameters(p, normalize=normalize, dim=1)
