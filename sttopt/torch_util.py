"""Conversion helpers at the boundary between torch tensors (`Problem`/`State`'s fields,
per `plans/torch_port_part2.md`'s Phase 3.1) and the NumPy/SciPy arrays that fixture
writing, `viz`, and CLI printing still speak.

A plain NumPy array and a torch tensor do not mix in an arithmetic expression (`ndarray
* tensor` raises `TypeError`, it does not silently upcast), so `optimize.step`/
`init_state` convert once at the boundary rather than relying on implicit interop.
"""

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor


def to_tensor(
    array: np.ndarray | Tensor,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """
    Move a NumPy array (or an existing tensor) onto `device` as a tensor.

    Passing a tensor through re-homes it rather than erroring, so a call site doesn't
    need to know whether it already has a tensor. `dtype` is optional and meant for
    floating-point fields: integer index/mask arrays should pass their own target dtype
    explicitly rather than being coerced to a real-valued `dtype`.

    :param array: source NumPy array or tensor.
    :param device: target device.
    :param dtype: target dtype; omitted to keep the source's own dtype.
    :return: tensor on `device` (and `dtype`, if given).
    """
    t = array if isinstance(array, Tensor) else torch.as_tensor(np.asarray(array))
    return t.to(device=device, dtype=dtype)


def to_tensors(
    arrays: dict[str, np.ndarray | Tensor],
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    """
    `to_tensor`, applied to every value in `arrays` at once.

    For a call site (e.g. `build_problem`) converting several same-dtype fields onto
    the same device in one boundary crossing -- keys match the returned dict's keys, so
    the result can be splatted straight into a dataclass constructor whose field names
    line up (`Problem(..., **to_tensors(...))`).

    :param arrays: mapping from field name to source array (or tensor).
    :param device: target device for every value.
    :param dtype: target dtype for every value; omitted to keep each source's own dtype.
    :return: mapping with the same keys, values converted.
    """
    return {name: to_tensor(a, device, dtype) for name, a in arrays.items()}


def csr_to_tensor(
    matrix: sp.spmatrix | sp.sparray, device: torch.device | str, dtype: torch.dtype
) -> Tensor:
    """
    Convert a SciPy sparse matrix to a `torch.sparse_csr_tensor` on `device`/`dtype`.

    :param matrix: any SciPy sparse matrix (converted to CSR first).
    :param device: target device.
    :param dtype: target dtype for the matrix's values.
    :return: equivalent `torch.sparse_csr_tensor`, on the host then moved to `device`.
    """
    csr = matrix.tocsr()
    # `torch.sparse_csr_tensor` requires sorted, duplicate-free column indices per row;
    # SciPy's own CSR does not guarantee that (confirmed on `filters.continuity_filter`'s
    # `L`, whose COO assembly interleaves several neighbor-offset passes unsorted).
    csr.sum_duplicates()
    csr.sort_indices()
    # Opt in explicitly (rather than let torch warn that it implicitly skipped them):
    # cheap here, since this runs once per `Problem`, not once per iteration.
    with torch.sparse.check_sparse_tensor_invariants():
        t = torch.sparse_csr_tensor(
            torch.as_tensor(csr.indptr, dtype=torch.int64),
            torch.as_tensor(csr.indices, dtype=torch.int64),
            torch.as_tensor(csr.data, dtype=dtype),
            size=csr.shape,
        )
    return t.to(device)


def to_numpy(x: Tensor | np.ndarray) -> np.ndarray:
    """
    Convert a dense tensor back to a NumPy array; pass a NumPy array through unchanged.

    For boundary code that stays plain-array by nature and isn't going to be ported:
    fixture writing, `viz`, CLI printing -- and for a SciPy sparse multiplication that
    needs a real array on its other side. `cli.py`/`viz.py`/fixture writing all need
    this to stay public.

    :param x: dense tensor or NumPy array.
    :return: NumPy array, detached and moved to host first if `x` was a tensor.
    """
    if isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def csr_to_scipy(x: Tensor) -> sp.csr_matrix:
    """
    Convert a `torch.sparse_csr_tensor` back to a SciPy `csr_matrix`, on the host.

    :param x: sparse CSR tensor.
    :return: equivalent SciPy `csr_matrix`.
    """
    return sp.csr_matrix(
        (
            x.values().detach().cpu().numpy(),
            x.col_indices().detach().cpu().numpy(),
            x.crow_indices().detach().cpu().numpy(),
        ),
        shape=tuple(x.shape),
    )
