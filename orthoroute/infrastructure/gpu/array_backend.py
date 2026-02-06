"""
Unified Array Backend - Abstraction layer for CuPy, MLX, and NumPy.

This module provides a unified interface (xp pattern) for array operations
that works across different backends:
- CuPy (NVIDIA CUDA)
- MLX (Apple Silicon Metal)
- NumPy (CPU fallback)

Usage:
    from orthoroute.infrastructure.gpu.array_backend import ArrayBackend, get_array_module

    backend = ArrayBackend.get_best_available()
    xp = backend.xp  # Array module (cupy, mlx.core, or numpy)

    # Use xp like numpy
    arr = xp.zeros((100, 100), dtype=xp.float32)
"""

import logging
from enum import Enum, auto
from typing import Any, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class BackendType(Enum):
    """Supported array backend types."""
    CUPY = auto()    # NVIDIA CUDA via CuPy
    MLX = auto()     # Apple Silicon via MLX
    NUMPY = auto()   # CPU fallback


# Detect available backends at import time
_CUPY_AVAILABLE = False
_MLX_AVAILABLE = False

try:
    import cupy as cp
    # Quick test
    _test = cp.array([1])
    _ = cp.sum(_test)
    _CUPY_AVAILABLE = True
    del _test
except (ImportError, Exception):
    cp = None

try:
    import mlx.core as mx
    # Quick test
    _test = mx.array([1])
    _ = mx.sum(_test)
    mx.eval(_)
    _MLX_AVAILABLE = True
    del _test
except (ImportError, Exception):
    mx = None


class ArrayBackend:
    """Unified array backend abstraction.

    Provides a consistent interface for array operations across
    CuPy (CUDA), MLX (Metal), and NumPy (CPU).

    Attributes:
        backend_type: The type of backend being used
        xp: The array module (cupy, mlx.core, or numpy)
        is_gpu: Whether this backend uses GPU acceleration
    """

    def __init__(self, backend_type: BackendType):
        """Initialize array backend.

        Args:
            backend_type: The backend type to use
        """
        self.backend_type = backend_type
        self._sparse = None

        if backend_type == BackendType.CUPY:
            if not _CUPY_AVAILABLE:
                raise RuntimeError("CuPy not available")
            self.xp = cp
            self.is_gpu = True
            self._name = "CuPy (CUDA)"
            import cupyx.scipy.sparse as sparse
            self._sparse = sparse

        elif backend_type == BackendType.MLX:
            if not _MLX_AVAILABLE:
                raise RuntimeError("MLX not available")
            self.xp = mx
            self.is_gpu = True
            self._name = "MLX (Metal)"
            # MLX doesn't have built-in sparse support yet
            self._sparse = None

        else:  # NUMPY
            self.xp = np
            self.is_gpu = False
            self._name = "NumPy (CPU)"
            from scipy import sparse
            self._sparse = sparse

    @property
    def name(self) -> str:
        """Get backend name."""
        return self._name

    @property
    def sparse(self):
        """Get sparse matrix module (if available)."""
        return self._sparse

    @classmethod
    def get_best_available(cls, prefer_gpu: bool = True) -> 'ArrayBackend':
        """Get the best available backend.

        Args:
            prefer_gpu: If True, prefer GPU backends over CPU

        Returns:
            ArrayBackend instance for the best available backend
        """
        if prefer_gpu:
            # Check for CUDA first (typically faster for this workload)
            if _CUPY_AVAILABLE:
                logger.info("Using CuPy (CUDA) backend")
                return cls(BackendType.CUPY)

            # Check for MLX (Apple Silicon)
            if _MLX_AVAILABLE:
                logger.info("Using MLX (Metal) backend")
                return cls(BackendType.MLX)

        # Fall back to NumPy
        logger.info("Using NumPy (CPU) backend")
        return cls(BackendType.NUMPY)

    @classmethod
    def get_apple_silicon_backend(cls) -> 'ArrayBackend':
        """Get backend optimized for Apple Silicon.

        Returns:
            MLX backend if available, otherwise NumPy
        """
        if _MLX_AVAILABLE:
            logger.info("Using MLX (Metal) backend for Apple Silicon")
            return cls(BackendType.MLX)

        logger.info("MLX not available, using NumPy (CPU) backend")
        return cls(BackendType.NUMPY)

    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert array to numpy (CPU).

        Args:
            array: Array from any backend

        Returns:
            numpy.ndarray
        """
        if isinstance(array, np.ndarray):
            return array

        if self.backend_type == BackendType.CUPY:
            return array.get()

        elif self.backend_type == BackendType.MLX:
            mx.eval(array)
            return np.array(array)

        return np.asarray(array)

    def from_numpy(self, array: np.ndarray) -> Any:
        """Convert numpy array to backend array.

        Args:
            array: numpy array

        Returns:
            Array in backend format
        """
        if self.backend_type == BackendType.CUPY:
            return cp.asarray(array)

        elif self.backend_type == BackendType.MLX:
            return mx.array(array)

        return array

    def synchronize(self) -> None:
        """Synchronize GPU operations (wait for completion)."""
        if self.backend_type == BackendType.CUPY:
            cp.cuda.Stream.null.synchronize()

        elif self.backend_type == BackendType.MLX:
            # Force evaluation of pending operations
            mx.eval(mx.array([0]))

        # NumPy is synchronous, nothing to do

    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Create zero-filled array."""
        if dtype is None:
            dtype = self.xp.float32 if self.backend_type != BackendType.NUMPY else np.float32
        return self.xp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Create one-filled array."""
        if dtype is None:
            dtype = self.xp.float32 if self.backend_type != BackendType.NUMPY else np.float32
        return self.xp.ones(shape, dtype=dtype)

    def full(self, shape: Tuple[int, ...], fill_value: Any, dtype=None) -> Any:
        """Create array filled with value."""
        if dtype is None:
            dtype = self.xp.float32 if self.backend_type != BackendType.NUMPY else np.float32

        if self.backend_type == BackendType.MLX:
            # Handle inf specially for MLX
            if fill_value == float('inf') or (hasattr(fill_value, 'item') and fill_value == np.inf):
                return mx.array(np.full(shape, np.inf, dtype=np.float32))
            return mx.full(shape, fill_value, dtype=dtype)

        return self.xp.full(shape, fill_value, dtype=dtype)

    def empty(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Create uninitialized array."""
        if dtype is None:
            dtype = self.xp.float32 if self.backend_type != BackendType.NUMPY else np.float32

        if self.backend_type == BackendType.MLX:
            # MLX doesn't have empty(), use zeros
            return mx.zeros(shape, dtype=dtype)

        return self.xp.empty(shape, dtype=dtype)

    def arange(self, *args, **kwargs) -> Any:
        """Create evenly spaced values."""
        return self.xp.arange(*args, **kwargs)

    def where(self, condition: Any, x: Any = None, y: Any = None) -> Any:
        """Return elements chosen from x or y depending on condition."""
        if x is None and y is None:
            # Return indices where condition is True
            if self.backend_type == BackendType.MLX:
                # MLX where() requires x and y, use numpy for index finding
                condition_np = self.to_numpy(condition)
                indices = np.where(condition_np)
                return tuple(mx.array(idx) for idx in indices)
            return self.xp.where(condition)

        return self.xp.where(condition, x, y)

    def sum(self, array: Any, axis: Optional[int] = None) -> Any:
        """Sum array elements."""
        return self.xp.sum(array, axis=axis)

    def min(self, array: Any, axis: Optional[int] = None) -> Any:
        """Minimum of array elements."""
        return self.xp.min(array, axis=axis)

    def max(self, array: Any, axis: Optional[int] = None) -> Any:
        """Maximum of array elements."""
        return self.xp.max(array, axis=axis)

    def argmin(self, array: Any, axis: Optional[int] = None) -> Any:
        """Index of minimum element."""
        return self.xp.argmin(array, axis=axis)

    def argmax(self, array: Any, axis: Optional[int] = None) -> Any:
        """Index of maximum element."""
        return self.xp.argmax(array, axis=axis)

    def abs(self, array: Any) -> Any:
        """Absolute value."""
        return self.xp.abs(array)

    def sqrt(self, array: Any) -> Any:
        """Square root."""
        return self.xp.sqrt(array)

    def clip(self, array: Any, a_min: Any, a_max: Any) -> Any:
        """Clip values to range."""
        return self.xp.clip(array, a_min, a_max)

    def concatenate(self, arrays: list, axis: int = 0) -> Any:
        """Join arrays along axis."""
        return self.xp.concatenate(arrays, axis=axis)

    def stack(self, arrays: list, axis: int = 0) -> Any:
        """Join arrays along new axis."""
        return self.xp.stack(arrays, axis=axis)

    def cumsum(self, array: Any, axis: Optional[int] = None) -> Any:
        """Cumulative sum."""
        return self.xp.cumsum(array, axis=axis)

    def unique(self, array: Any) -> Any:
        """Find unique elements."""
        if self.backend_type == BackendType.MLX:
            # MLX may not have unique, fall back to numpy
            arr_np = self.to_numpy(array)
            return mx.array(np.unique(arr_np))
        return self.xp.unique(array)

    def isinf(self, array: Any) -> Any:
        """Test for infinity."""
        return self.xp.isinf(array)

    def isnan(self, array: Any) -> Any:
        """Test for NaN."""
        return self.xp.isnan(array)

    def logical_and(self, x1: Any, x2: Any) -> Any:
        """Element-wise logical AND."""
        return self.xp.logical_and(x1, x2)

    def logical_or(self, x1: Any, x2: Any) -> Any:
        """Element-wise logical OR."""
        return self.xp.logical_or(x1, x2)

    def logical_not(self, array: Any) -> Any:
        """Element-wise logical NOT."""
        return self.xp.logical_not(array)

    def astype(self, array: Any, dtype) -> Any:
        """Cast array to a different type."""
        if self.backend_type == BackendType.MLX:
            return array.astype(dtype)
        return array.astype(dtype)

    def count_nonzero(self, array: Any) -> int:
        """Count non-zero elements."""
        if self.backend_type == BackendType.MLX:
            # MLX may not have count_nonzero
            return int(mx.sum(array != 0))
        return int(self.xp.count_nonzero(array))

    def scatter_min(self, array: Any, indices: Any, values: Any) -> Any:
        """Atomic minimum scatter operation.

        This is equivalent to CuPy's minimum.at() or indexed minimum assignment.
        For each index i in indices, sets array[indices[i]] = min(array[indices[i]], values[i])

        Note: This is a complex operation that may not be fully supported on all backends.
        Falls back to a loop-based implementation for non-CUDA backends.
        """
        if self.backend_type == BackendType.CUPY:
            cp.minimum.at(array, indices, values)
            return array

        elif self.backend_type == BackendType.MLX:
            # MLX doesn't have scatter_min, implement via numpy and convert
            arr_np = self.to_numpy(array)
            idx_np = self.to_numpy(indices)
            val_np = self.to_numpy(values)
            np.minimum.at(arr_np, idx_np, val_np)
            return mx.array(arr_np)

        else:  # NumPy
            np.minimum.at(array, indices, values)
            return array

    def scatter_add(self, array: Any, indices: Any, values: Any) -> Any:
        """Atomic addition scatter operation.

        This is equivalent to CuPy's add.at() or indexed addition.
        For each index i in indices, sets array[indices[i]] += values[i]
        """
        if self.backend_type == BackendType.CUPY:
            cp.add.at(array, indices, values)
            return array

        elif self.backend_type == BackendType.MLX:
            # MLX doesn't have scatter_add, implement via numpy and convert
            arr_np = self.to_numpy(array)
            idx_np = self.to_numpy(indices)
            val_np = self.to_numpy(values)
            np.add.at(arr_np, idx_np, val_np)
            return mx.array(arr_np)

        else:  # NumPy
            np.add.at(array, indices, values)
            return array

    def create_csr_matrix(self, data: Any, indices: Any, indptr: Any, shape: Tuple[int, int]):
        """Create a CSR sparse matrix.

        Args:
            data: Non-zero values
            indices: Column indices
            indptr: Row pointers
            shape: Matrix shape (rows, cols)

        Returns:
            CSR sparse matrix in backend-appropriate format
        """
        if self.backend_type == BackendType.CUPY:
            import cupyx.scipy.sparse as sparse
            return sparse.csr_matrix((data, indices, indptr), shape=shape)

        elif self.backend_type == BackendType.MLX:
            # MLX doesn't have native sparse support
            # Return a dict-based representation that can be used for CSR operations
            return {
                'data': data,
                'indices': indices,
                'indptr': indptr,
                'shape': shape,
                'format': 'csr'
            }

        else:  # NumPy
            from scipy.sparse import csr_matrix
            return csr_matrix((data, indices, indptr), shape=shape)


def get_array_module(prefer_gpu: bool = True):
    """Get array module (xp) from best available backend.

    This is a convenience function for quick access to the array module.

    Args:
        prefer_gpu: If True, prefer GPU backends

    Returns:
        Array module (cupy, mlx.core, or numpy)
    """
    backend = ArrayBackend.get_best_available(prefer_gpu=prefer_gpu)
    return backend.xp


def get_backend(prefer_gpu: bool = True) -> ArrayBackend:
    """Get best available array backend.

    Args:
        prefer_gpu: If True, prefer GPU backends

    Returns:
        ArrayBackend instance
    """
    return ArrayBackend.get_best_available(prefer_gpu=prefer_gpu)


# Module-level detection results
def is_cupy_available() -> bool:
    """Check if CuPy is available."""
    return _CUPY_AVAILABLE


def is_mlx_available() -> bool:
    """Check if MLX is available."""
    return _MLX_AVAILABLE


def get_available_backends() -> list:
    """Get list of available backends."""
    available = [BackendType.NUMPY]  # Always available
    if _CUPY_AVAILABLE:
        available.insert(0, BackendType.CUPY)
    if _MLX_AVAILABLE:
        available.insert(0 if not _CUPY_AVAILABLE else 1, BackendType.MLX)
    return available
