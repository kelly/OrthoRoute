"""GPU infrastructure adapters.

Provides GPU acceleration via multiple backends:
- CuPy (NVIDIA CUDA)
- MLX (Apple Silicon Metal)
- NumPy (CPU fallback)
"""
from .cuda_provider import CUDAProvider
from .cpu_fallback import CPUFallbackProvider, CPUProvider
from .mlx_provider import MLXProvider, MLX_AVAILABLE
from .array_backend import (
    ArrayBackend,
    BackendType,
    get_array_module,
    get_backend,
    is_cupy_available,
    is_mlx_available,
    get_available_backends,
)

__all__ = [
    # Providers
    'CUDAProvider',
    'CPUFallbackProvider',
    'CPUProvider',
    'MLXProvider',
    # Array backend abstraction
    'ArrayBackend',
    'BackendType',
    'get_array_module',
    'get_backend',
    # Detection
    'is_cupy_available',
    'is_mlx_available',
    'get_available_backends',
    'MLX_AVAILABLE',
]


def get_best_provider():
    """Get the best available GPU provider for the current system.

    Returns:
        GPUProvider: CUDAProvider, MLXProvider, or CPUFallbackProvider
    """
    # Try CUDA first (typically faster for this workload)
    try:
        cuda = CUDAProvider()
        if cuda.is_available():
            return cuda
    except Exception:
        pass

    # Try MLX (Apple Silicon)
    try:
        mlx = MLXProvider()
        if mlx.is_available():
            return mlx
    except Exception:
        pass

    # Fall back to CPU
    return CPUFallbackProvider()