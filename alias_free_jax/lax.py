from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
from jax import lax

Array = Any
DType = Any


def conv_dimension_numbers(input_shape):
    """Computes the dimension numbers based on the input shape."""
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


def conv_transpose_padding(k, s, padding):
    if padding == "SAME":
        pad_len = k + s - 2
        if s > k - 1:
            pad_a = k - 1
        else:
            pad_a = int(np.ceil(pad_len / 2))
    elif padding == "VALID":
        pad_len = k + s - 2 + max(k - s, 0)
        pad_a = k - 1
    else:
        raise ValueError("Padding mode must be `SAME` or `VALID`.")
    pad_b = pad_len - pad_a
    return pad_a, pad_b


def conv_transpose(
    lhs: Array,
    rhs: Array,
    strides: Sequence[int],
    padding: Union[str, Sequence[Tuple[int, int]]],
    rhs_dilation: Optional[Sequence[int]] = None,
    feature_group_count: int = 1,
) -> Array:
    assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
    ndims = len(lhs.shape)
    one = (1,) * (ndims - 2)
    # Set dimensional layout defaults if not specified.
    if ndims == 2:
        dimension_numbers = ("NC", "IO", "NC")
    elif ndims == 3:
        dimension_numbers = ("NHC", "HIO", "NHC")
    elif ndims == 4:
        dimension_numbers = ("NHWC", "HWIO", "NHWC")
    elif ndims == 5:
        dimension_numbers = ("NHWDC", "HWDIO", "NHWDC")
    else:
        raise ValueError("No 4+ dimensional dimension_number defaults.")
    dn = lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    k_shape = np.take(rhs.shape, dn.rhs_spec)
    k_sdims = k_shape[2:]  # type: ignore[index]
    # Calculate correct output shape given padding and strides.
    pads: Union[str, Sequence[Tuple[int, int]]]
    if isinstance(padding, str) and padding in {"SAME", "VALID"}:
        rhs_dilation = (1,) * (rhs.ndim - 2)
        effective_k_size = map(lambda k, r: (k - 1) * r + 1, k_sdims, rhs_dilation)
        pads = [
            conv_transpose_padding(k, s, padding)
            for k, s in zip(effective_k_size, strides)
        ]
    else:
        pads = padding
    return lax.conv_general_dilated(
        lhs, rhs, one, pads, strides, rhs_dilation, dn, feature_group_count
    )
