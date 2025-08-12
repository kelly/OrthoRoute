#!/usr/bin/env python3
"""
GPU Detection and Status for OrthoRoute
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class GPUStatus:
    """Detect and report GPU availability for routing acceleration"""
    
    def __init__(self):
        self.gpu_available = False
        self.gpu_name = "None"
        self.gpu_memory = 0
        self.gpu_memory_str = "0GB"
        self.cupy_available = False
        self.cuda_version = "N/A"
        self._detect_gpu()
    
    def _detect_gpu(self):
        """Detect available GPU hardware and CuPy support"""
        try:
            # Try to import CuPy for CUDA support
            import cupy as cp
            self.cupy_available = True
            
            # Get GPU device info
            device = cp.cuda.Device()
            
            # Get GPU name using device properties
            try:
                # Try to get device name
                device_props = cp.cuda.runtime.getDeviceProperties(device.id)
                self.gpu_name = device_props['name'].decode('utf-8') if isinstance(device_props['name'], bytes) else str(device_props['name'])
            except Exception:
                # Fallback to generic name
                self.gpu_name = f"CUDA Device {device.id}"
            
            # Get GPU memory
            meminfo = cp.cuda.runtime.memGetInfo()
            self.gpu_memory = meminfo[1]  # Total memory in bytes
            self.gpu_memory_str = f"{self.gpu_memory / (1024**3):.1f}GB"
            
            # Get CUDA version
            cuda_version = cp.cuda.runtime.runtimeGetVersion()
            major = cuda_version // 1000
            minor = (cuda_version % 1000) // 10
            self.cuda_version = f"{major}.{minor}"
            
            self.gpu_available = True
            
            logger.info(f"✅ GPU detected: {self.gpu_name} ({self.gpu_memory_str})")
            logger.info(f"   CUDA version: {self.cuda_version}")
            
        except ImportError:
            logger.info("CuPy not available - using CPU fallback")
            self._set_cpu_fallback("CuPy not installed")
            
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
            self._set_cpu_fallback(f"GPU error: {e}")
    
    def _set_cpu_fallback(self, reason: str):
        """Set CPU fallback mode"""
        self.gpu_available = False
        self.gpu_name = f"CPU Fallback ({reason})"
        self.gpu_memory = 0
        self.gpu_memory_str = "System RAM"
        self.cupy_available = False
        self.cuda_version = "N/A"
    
    def get_status_string(self) -> str:
        """Get formatted status string for UI display"""
        if self.gpu_available:
            return f"GPU Ready: {self.gpu_name} - {self.gpu_memory_str} VRAM"
        else:
            return f"CPU Fallback: {self.gpu_name}"
    
    def get_detailed_info(self) -> Dict:
        """Get detailed GPU information"""
        return {
            'available': self.gpu_available,
            'name': self.gpu_name,
            'memory_gb': self.gpu_memory / (1024**3) if self.gpu_memory > 0 else 0,
            'memory_str': self.gpu_memory_str,
            'cupy_available': self.cupy_available,
            'cuda_version': self.cuda_version,
            'status': self.get_status_string()
        }
    
    def can_use_gpu_routing(self) -> bool:
        """Check if GPU routing is available"""
        return self.gpu_available and self.cupy_available
    
    def get_recommended_algorithm(self) -> str:
        """Get recommended routing algorithm based on hardware"""
        if self.can_use_gpu_routing():
            return "auto"  # GPU-accelerated auto mode
        else:
            return "frontier_reduction"  # CPU fallback


# Global GPU status instance
_gpu_status = None

def get_gpu_status() -> GPUStatus:
    """Get singleton GPU status instance"""
    global _gpu_status
    if _gpu_status is None:
        _gpu_status = GPUStatus()
    return _gpu_status


def test_gpu_detection():
    """Test GPU detection functionality"""
    print("🔍 Testing GPU Detection...")
    print("=" * 40)
    
    status = get_gpu_status()
    info = status.get_detailed_info()
    
    print(f"GPU Available: {info['available']}")
    print(f"GPU Name: {info['name']}")
    print(f"GPU Memory: {info['memory_str']}")
    print(f"CuPy Available: {info['cupy_available']}")
    print(f"CUDA Version: {info['cuda_version']}")
    print(f"Status: {info['status']}")
    print(f"Can Use GPU Routing: {status.can_use_gpu_routing()}")
    print(f"Recommended Algorithm: {status.get_recommended_algorithm()}")
    
    return status


if __name__ == "__main__":
    test_gpu_detection()
