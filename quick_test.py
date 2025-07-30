#!/usr/bin/env python3
"""
Quick validation test for OrthoRoute core components
"""

import sys
import os

def test_basic_imports():
    """Test basic Python imports work"""
    print("🔧 Testing basic imports...")
    try:
        import json
        import math
        import random
        print("✅ Standard library imports OK")
        return True
    except Exception as e:
        print(f"❌ Basic imports failed: {e}")
        return False

def test_optional_gpu():
    """Test GPU acceleration (optional)"""
    print("🚀 Testing GPU acceleration...")
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props["name"].decode("utf-8")
        print(f"✅ GPU detected: {gpu_name}")
        return True
    except ImportError:
        print("⚠️  CuPy not available - CPU mode only")
        return True  # This is OK
    except Exception as e:
        print(f"⚠️  GPU issue (CPU fallback available): {e}")
        return True  # This is OK

def test_addon_structure():
    """Test addon package structure"""
    print("📦 Testing addon package structure...")
    
    addon_dir = "addon_package/plugins"
    required_files = [
        "__init__.py",
        "orthoroute_engine.py",
        "api_bridge.py",
        "icon.png"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(addon_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file)
        else:
            size = os.path.getsize(file_path)
            print(f"  ✓ {file} ({size} bytes)")
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def test_package_size():
    """Test package size is reasonable"""
    print("📏 Testing package size...")
    
    zip_path = "orthoroute-kicad-addon.zip"
    if os.path.exists(zip_path):
        size_kb = os.path.getsize(zip_path) / 1024
        print(f"  Package size: {size_kb:.1f} KB")
        
        if size_kb > 100:
            print(f"⚠️  Package is large ({size_kb:.1f} KB) - consider optimization")
        else:
            print("✅ Package size is reasonable")
        return True
    else:
        print("❌ Package file not found - run 'python build_addon.py' first")
        return False

def test_installation():
    """Test if development installation exists"""
    print("🔌 Testing installation...")
    
    # Check typical KiCad plugin directories
    possible_dirs = [
        os.path.expanduser("~/Documents/KiCad/9.0/scripting/plugins/OrthoRoute"),
        os.path.expanduser("~/.kicad/scripting/plugins/OrthoRoute"),
        os.path.join(os.getenv("APPDATA", ""), "kicad", "9.0", "scripting", "plugins", "OrthoRoute"),
        "c:\\Users\\Benchoff\\Documents\\KiCad\\9.0\\scripting\\plugins\\OrthoRoute"
    ]
    
    for plugin_dir in possible_dirs:
        if os.path.exists(plugin_dir):
            files = os.listdir(plugin_dir)
            print(f"✅ Installation found: {plugin_dir}")
            print(f"  Files: {len(files)} ({', '.join(files[:3])}{'...' if len(files) > 3 else ''})")
            return True
    
    print("⚠️  No installation found - run 'python install_dev.py' first")
    return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 OrthoRoute Quick Validation Test")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_optional_gpu,
        test_addon_structure,
        test_package_size,
        test_installation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
        print()
    
    print("=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    
    test_names = [
        "Basic Imports",
        "GPU Acceleration", 
        "Package Structure",
        "Package Size",
        "Installation"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED - OrthoRoute is ready!")
        print("\nNext steps:")
        print("1. Open KiCad PCB Editor")
        print("2. Look for 'OrthoRoute GPU Autorouter' in Tools → External Plugins")
        print("3. Load a PCB with unrouted nets and test the plugin")
    elif passed >= len(tests) - 1:
        print("✅ MOSTLY READY - Minor issues that shouldn't prevent usage")
    else:
        print("⚠️  ISSUES DETECTED - Check failures above")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
