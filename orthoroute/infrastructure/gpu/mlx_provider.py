"""MLX GPU provider implementation for Apple Silicon."""
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ...application.interfaces.gpu_provider import GPUProvider

logger = logging.getLogger(__name__)

# Try to import MLX
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    mx = None
    MLX_AVAILABLE = False


class MLXProvider(GPUProvider):
    """MLX/Metal implementation of GPU provider for Apple Silicon.

    Uses Apple's MLX framework for GPU-accelerated array operations
    on M-series chips. Provides similar functionality to CuPy but
    targets Metal instead of CUDA.
    """

    def __init__(self):
        """Initialize MLX provider."""
        self._initialized = False
        self._device_info = {}
        self._memory_limit = None

    def is_available(self) -> bool:
        """Check if MLX is available on this system."""
        if not MLX_AVAILABLE:
            logger.warning("MLX not installed - Apple Silicon GPU acceleration unavailable")
            return False

        try:
            # Test basic MLX functionality
            test_array = mx.array([1, 2, 3])
            _ = mx.sum(test_array)
            mx.eval(_)  # Force computation
            logger.info("MLX (Apple Silicon GPU) detected and working")
            return True
        except Exception as e:
            logger.warning(f"MLX error: {str(e)} - Apple Silicon GPU acceleration unavailable")
            return False

    def initialize(self) -> bool:
        """Initialize MLX resources."""
        if self._initialized:
            return True

        if not MLX_AVAILABLE:
            logger.error("MLX not available - cannot initialize")
            return False

        try:
            # MLX doesn't require explicit initialization like CUDA
            # but we can set up some configuration

            # Get device information (MLX uses unified memory on Apple Silicon)
            import platform
            import subprocess

            # Try to get chip info
            try:
                result = subprocess.run(
                    ['sysctl', '-n', 'machdep.cpu.brand_string'],
                    capture_output=True, text=True
                )
                chip_name = result.stdout.strip() if result.returncode == 0 else 'Apple Silicon'
            except Exception:
                chip_name = 'Apple Silicon'

            # Get memory info (unified memory on Apple Silicon)
            try:
                import psutil
                total_mem = psutil.virtual_memory().total
                available_mem = psutil.virtual_memory().available
            except ImportError:
                total_mem = 0
                available_mem = 0

            # Set memory limit (70% of available for safety with unified memory)
            self._memory_limit = int(available_mem * 0.7)

            self._device_info = {
                'name': chip_name,
                'compute_capability': 'Metal',
                'total_memory': total_mem,
                'free_memory': available_mem,
                'memory_limit': self._memory_limit,
                'device_id': 'mlx_gpu',
                'backend': 'MLX'
            }

            # Test basic operations
            test_array = mx.ones((100, 100), dtype=mx.float32)
            result = mx.sum(test_array)
            mx.eval(result)

            self._initialized = True
            logger.info(f"MLX initialized: {chip_name}")
            logger.info(f"Unified Memory: {total_mem / 1024**3:.1f}GB total, {available_mem / 1024**3:.1f}GB available")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize MLX: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup MLX resources."""
        # MLX handles memory automatically through Python garbage collection
        # and Metal's unified memory architecture
        self._initialized = False
        logger.debug("MLX provider cleaned up")

    def get_device_info(self) -> Dict[str, Any]:
        """Get MLX device information."""
        return self._device_info.copy()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        if not self._initialized:
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }

        try:
            import psutil
            vm = psutil.virtual_memory()

            return {
                'total_memory': vm.total,
                'free_memory': vm.available,
                'used_memory': vm.used,
                # MLX uses unified memory, so pool tracking is approximate
                'memory_pool_used': vm.used,
                'memory_pool_total': vm.total
            }

        except Exception as e:
            logger.error(f"Error getting MLX memory info: {e}")
            return {
                'total_memory': 0,
                'free_memory': 0,
                'used_memory': 0,
                'memory_pool_used': 0,
                'memory_pool_total': 0
            }

    def create_array(self, shape: Tuple[int, ...], dtype=None, fill_value=None) -> Any:
        """Create array on MLX (GPU)."""
        if not self._initialized:
            raise RuntimeError("MLX not initialized")

        # Map numpy dtypes to MLX dtypes
        if dtype is None:
            mlx_dtype = mx.float32
        else:
            mlx_dtype = self._numpy_to_mlx_dtype(dtype)

        try:
            if fill_value is None:
                # MLX doesn't have empty(), use zeros as closest equivalent
                arr = mx.zeros(shape, dtype=mlx_dtype)
            elif fill_value == 0:
                arr = mx.zeros(shape, dtype=mlx_dtype)
            elif fill_value == 1:
                arr = mx.ones(shape, dtype=mlx_dtype)
            elif fill_value == float('inf') or (hasattr(fill_value, 'item') and fill_value == np.inf):
                # MLX doesn't have full() with inf, use numpy then convert
                arr = mx.array(np.full(shape, np.inf, dtype=np.float32))
            else:
                arr = mx.full(shape, fill_value, dtype=mlx_dtype)

            return arr

        except Exception as e:
            logger.error(f"Error creating MLX array: {e}")
            raise

    def copy_array(self, array: Any) -> Any:
        """Create copy of array on MLX."""
        if not self._initialized:
            raise RuntimeError("MLX not initialized")

        try:
            # MLX arrays are immutable, so we create a copy by adding zero
            # or use array() constructor
            if hasattr(array, '__mlx_array__') or str(type(array).__module__).startswith('mlx'):
                return mx.array(array)
            else:
                # Convert from numpy first
                return mx.array(np.asarray(array))
        except Exception as e:
            logger.error(f"Error copying MLX array: {e}")
            raise

    def to_cpu(self, array: Any) -> np.ndarray:
        """Convert MLX array to CPU (numpy)."""
        if not self._initialized:
            return array if isinstance(array, np.ndarray) else np.asarray(array)

        try:
            if hasattr(array, '__mlx_array__') or str(type(array).__module__).startswith('mlx'):
                # Ensure computation is complete
                mx.eval(array)
                return np.array(array)
            else:
                return array if isinstance(array, np.ndarray) else np.asarray(array)
        except Exception as e:
            logger.error(f"Error converting MLX array to CPU: {e}")
            return array if isinstance(array, np.ndarray) else np.asarray(array)

    def to_gpu(self, array: np.ndarray) -> Any:
        """Convert CPU array to MLX (GPU)."""
        if not self._initialized:
            raise RuntimeError("MLX not initialized")

        try:
            return mx.array(array)
        except Exception as e:
            logger.error(f"Error converting array to MLX: {e}")
            raise

    def synchronize(self) -> None:
        """Synchronize MLX operations (wait for GPU completion)."""
        if not self._initialized:
            return

        try:
            # MLX uses lazy evaluation, eval() forces computation
            # There's no global sync, but we can create a dummy op
            mx.eval(mx.array([0]))
        except Exception as e:
            logger.error(f"Error synchronizing MLX: {e}")

    def _numpy_to_mlx_dtype(self, dtype):
        """Convert numpy dtype to MLX dtype."""
        if dtype is None:
            return mx.float32

        # Handle numpy dtype objects and strings
        dtype_str = str(np.dtype(dtype))

        dtype_map = {
            'float32': mx.float32,
            'float64': mx.float32,  # MLX prefers float32, downcast float64
            'float16': mx.float16,
            'int32': mx.int32,
            'int64': mx.int64,
            'int16': mx.int16,
            'int8': mx.int8,
            'uint8': mx.uint8,
            'uint16': mx.uint16,
            'uint32': mx.uint32,
            'uint64': mx.uint64,
            'bool': mx.bool_,
        }

        return dtype_map.get(dtype_str, mx.float32)

    def __enter__(self):
        """Context manager entry."""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


# Convenience function to get the best available provider
def get_best_provider() -> GPUProvider:
    """Get the best available GPU provider for the current system.

    Returns MLX on Apple Silicon, falls back to CPU otherwise.
    """
    mlx_provider = MLXProvider()
    if mlx_provider.is_available():
        return mlx_provider

    # Fall back to CPU
    from .cpu_fallback import CPUFallbackProvider
    return CPUFallbackProvider()
