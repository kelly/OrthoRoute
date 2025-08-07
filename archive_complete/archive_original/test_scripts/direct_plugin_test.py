#!/usr/bin/env python3
"""
Direct Plugin Test - No KiCad GUI Required
=========================================
Tests the core routing functionality without any UI dependencies
"""

import sys
import os
import traceback
import datetime

def main():
    """Test the plugin core functionality"""
    print("🔍 TESTING ORTHOROUTE CORE FUNCTIONALITY")
    print("=" * 50)
    
    # Create comprehensive debug log
    desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(desktop_path, f"OrthoRoute_Direct_Test_{timestamp}.txt")
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        def log_print(msg):
            print(msg)
            log_file.write(msg + "\n")
            log_file.flush()
        
        log_print(f"Direct Plugin Test - {datetime.datetime.now()}")
        log_print("=" * 60)
        
        try:
            # Test 1: Basic Python environment
            log_print("\n🐍 Testing Python Environment:")
            log_print(f"Python version: {sys.version}")
            log_print(f"Python executable: {sys.executable}")
            log_print(f"Current working directory: {os.getcwd()}")
            
            # Test 2: Path setup (same as plugin does)
            log_print("\n🔧 Setting up paths (as plugin does):")
            original_path = sys.path.copy()
            
            system_site_packages = [
                r"C:\Users\Benchoff\AppData\Roaming\Python\Python312\site-packages",
                r"C:\Python312\Lib\site-packages"
            ]
            
            paths_added = 0
            for path in system_site_packages:
                if os.path.exists(path):
                    if path not in sys.path:
                        sys.path.insert(0, path)
                        paths_added += 1
                        log_print(f"  ✓ Added: {path}")
                    else:
                        log_print(f"  ✓ Already present: {path}")
                else:
                    log_print(f"  ❌ Path doesn't exist: {path}")
            
            log_print(f"Paths added: {paths_added}, Total paths: {len(sys.path)}")
            
            # Test 3: CuPy import (core issue)
            log_print("\n🚀 Testing CuPy Import:")
            try:
                import cupy as cp
                device = cp.cuda.Device()
                props = cp.cuda.runtime.getDeviceProperties(device.id)
                gpu_name = props["name"].decode("utf-8")
                log_print(f"✅ CuPy SUCCESS: {gpu_name}")
                log_print(f"✅ CuPy version: {cp.__version__}")
                cupy_available = True
            except Exception as e:
                log_print(f"❌ CuPy FAILED: {e}")
                log_print("📋 Full CuPy error:")
                log_print(traceback.format_exc())
                cupy_available = False
            
            # Test 4: Core algorithm test (without pcbnew)
            log_print("\n🧮 Testing Core Algorithm (No pcbnew):")
            if cupy_available:
                try:
                    import cupy as cp
                    
                    # Test basic GPU operations
                    log_print("  Testing basic GPU operations...")
                    test_array = cp.array([1, 2, 3, 4, 5])
                    result = cp.sum(test_array)
                    log_print(f"  ✅ GPU array sum: {result}")
                    
                    # Test grid operations (core of routing)
                    log_print("  Testing grid operations...")
                    grid_size = (100, 100, 2)  # Small test grid
                    grid = cp.zeros(grid_size, dtype=cp.int32)
                    log_print(f"  ✅ Created GPU grid: {grid.shape}")
                    
                    # Test wavefront expansion (simplified)
                    log_print("  Testing wavefront expansion...")
                    start_pos = (50, 50, 0)
                    grid[start_pos] = 1
                    neighbors = cp.array([
                        [start_pos[0]-1, start_pos[1], start_pos[2]],
                        [start_pos[0]+1, start_pos[1], start_pos[2]],
                        [start_pos[0], start_pos[1]-1, start_pos[2]],
                        [start_pos[0], start_pos[1]+1, start_pos[2]]
                    ])
                    log_print(f"  ✅ Neighbors calculated: {neighbors.shape}")
                    
                    # Test memory allocation for larger grids
                    log_print("  Testing larger grid allocation...")
                    large_grid = cp.zeros((1000, 1000, 4), dtype=cp.int32)
                    log_print(f"  ✅ Large grid created: {large_grid.shape}")
                    del large_grid  # Free memory
                    
                    log_print("✅ Core algorithm tests PASSED")
                    
                except Exception as e:
                    log_print(f"❌ Core algorithm test FAILED: {e}")
                    log_print(traceback.format_exc())
            
            # Test 5: Plugin file access
            log_print("\n📁 Testing Plugin File Access:")
            plugin_file = os.path.join("addon_package", "plugins", "__init__.py")
            if os.path.exists(plugin_file):
                file_size = os.path.getsize(plugin_file)
                log_print(f"✅ Plugin file found: {plugin_file} ({file_size} bytes)")
                
                # Try to read the plugin code
                try:
                    with open(plugin_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    log_print(f"✅ Plugin file readable: {len(content)} characters")
                    
                    # Check for key functions
                    if "_route_board_gpu" in content:
                        log_print("✅ GPU routing function found")
                    else:
                        log_print("❌ GPU routing function NOT found")
                    
                    if "class OrthoRouteKiCadPlugin" in content:
                        log_print("✅ Plugin class found")
                    else:
                        log_print("❌ Plugin class NOT found")
                        
                except Exception as e:
                    log_print(f"❌ Could not read plugin file: {e}")
            else:
                log_print(f"❌ Plugin file NOT found: {plugin_file}")
            
            # Test 6: Direct algorithm simulation
            log_print("\n🎯 Direct Algorithm Simulation:")
            if cupy_available:
                try:
                    log_print("  Simulating PCB routing scenario...")
                    
                    # Create mock PCB grid (100mm x 100mm, 0.1mm resolution, 2 layers)
                    grid_width = 1000  # 100mm / 0.1mm
                    grid_height = 1000
                    layers = 2
                    
                    log_print(f"  Creating mock PCB grid: {grid_width}x{grid_height}x{layers}")
                    
                    # Test memory allocation
                    pcb_grid = cp.zeros((grid_width, grid_height, layers), dtype=cp.int32)
                    distance_grid = cp.full((grid_width, grid_height, layers), -1, dtype=cp.int32)
                    
                    log_print("  ✅ PCB grids allocated successfully")
                    
                    # Mock routing: point A to point B
                    start_x, start_y = 100, 100  # 10mm, 10mm
                    end_x, end_y = 900, 900      # 90mm, 90mm
                    layer = 0
                    
                    log_print(f"  Mock route: ({start_x},{start_y}) to ({end_x},{end_y}) on layer {layer}")
                    
                    # Set start point
                    distance_grid[start_x, start_y, layer] = 0
                    queue = [(start_x, start_y, layer)]
                    
                    # Simple BFS simulation (first few steps)
                    steps = 0
                    max_steps = 1000
                    found_target = False
                    
                    directions = [(-1,0,0), (1,0,0), (0,-1,0), (0,1,0)]  # 4-directional
                    
                    while queue and steps < max_steps:
                        x, y, z = queue.pop(0)
                        current_dist = distance_grid[x, y, z]
                        
                        if x == end_x and y == end_y:
                            found_target = True
                            log_print(f"  ✅ Target found in {steps} steps, distance {current_dist}")
                            break
                        
                        for dx, dy, dz in directions:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            
                            if (0 <= nx < grid_width and 0 <= ny < grid_height and 
                                0 <= nz < layers and distance_grid[nx, ny, nz] == -1):
                                distance_grid[nx, ny, nz] = current_dist + 1
                                queue.append((nx, ny, nz))
                        
                        steps += 1
                    
                    if found_target:
                        log_print("✅ Mock routing algorithm SUCCESSFUL")
                    else:
                        log_print(f"⚠️  Mock routing incomplete after {steps} steps")
                    
                    log_print("✅ Algorithm simulation completed")
                    
                except Exception as e:
                    log_print(f"❌ Algorithm simulation FAILED: {e}")
                    log_print(traceback.format_exc())
                    
        except Exception as e:
            log_print(f"❌ CRITICAL TEST FAILURE: {e}")
            log_print(traceback.format_exc())
        
        log_print("\n" + "=" * 60)
        log_print("TEST COMPLETE")
        log_print(f"Log saved to: {log_path}")
        log_print("=" * 60)
    
    print(f"\n📄 Detailed log saved to: {log_path}")
    print("\nThis test runs the core plugin functionality without KiCad.")
    print("Check the log file to see exactly where the plugin might be failing!")

if __name__ == "__main__":
    main()
