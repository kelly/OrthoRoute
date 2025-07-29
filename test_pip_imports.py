"""
Quick test of import strategy after pip installation
"""
import sys
print("Testing import strategy...")

try:
    from orthoroute.standalone_wave_router import WaveRouter
    print("✓ SUCCESS: Imported WaveRouter from orthoroute.standalone_wave_router")
except ImportError as e:
    print(f"✗ FAILED: Could not import from orthoroute.standalone_wave_router: {e}")

try:
    from orthoroute.gpu_engine import OrthoRouteEngine
    print("✓ SUCCESS: Imported OrthoRouteEngine from orthoroute.gpu_engine")
    
    # Test engine creation
    engine = OrthoRouteEngine()
    print(f"✓ SUCCESS: Created engine with ID: {engine.engine_id}")
    
except ImportError as e:
    print(f"✗ FAILED: Could not import OrthoRouteEngine: {e}")
except Exception as e:
    print(f"✗ FAILED: Error creating engine: {e}")

print("Import strategy test complete!")
