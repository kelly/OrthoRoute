#!/usr/bin/env python3
"""
OrthoRoute Headless Test Script
Tests OrthoRoute functionality using kicad-cli in headless mode
"""

import sys
import os
import json
import traceback
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "addon_package" / "plugins"))

def test_orthoroute_headless():
    """Main headless test function"""
    print("=" * 80)
    print("OrthoRoute Headless Test Suite")
    print("=" * 80)
    
    test_results = {
        'import_tests': {},
        'api_tests': {},
        'board_tests': {},
        'routing_tests': {},
        'overall_success': False
    }
    
    try:
        # Test 1: Import Tests
        print("\n1. Testing Imports...")
        test_results['import_tests'] = test_imports()
        
        # Test 2: API Tests
        print("\n2. Testing KiCad API...")
        test_results['api_tests'] = test_kicad_api()
        
        # Test 3: Board Tests (if board available)
        print("\n3. Testing Board Operations...")
        test_results['board_tests'] = test_board_operations()
        
        # Test 4: Routing Engine Tests
        print("\n4. Testing Routing Engine...")
        test_results['routing_tests'] = test_routing_engine()
        
        # Overall assessment
        test_results['overall_success'] = assess_overall_success(test_results)
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR in test suite: {e}")
        traceback.print_exc()
        test_results['critical_error'] = str(e)
    
    # Print results summary
    print_test_summary(test_results)
    
    # Save results to file
    save_test_results(test_results)
    
    return test_results

def test_imports():
    """Test all required imports"""
    import_results = {}
    
    # Test pcbnew import
    try:
        import pcbnew
        import_results['pcbnew'] = {'success': True, 'version': 'Available'}
        print("pcbnew imported successfully")
    except ImportError as e:
        import_results['pcbnew'] = {'success': False, 'error': str(e)}
        print(f"pcbnew import failed: {e}")
    
    # Test wx import
    try:
        import wx
        import_results['wx'] = {'success': True, 'version': wx.version()}
        print(f"wx imported successfully (version: {wx.version()})")
    except ImportError as e:
        import_results['wx'] = {'success': False, 'error': str(e)}
        print(f"wx import failed: {e}")
    
    # Test OrthoRoute engine import
    try:
        from orthoroute_engine import OrthoRouteEngine
        import_results['orthoroute_engine'] = {'success': True}
        print("OrthoRoute engine imported successfully")
    except ImportError as e:
        import_results['orthoroute_engine'] = {'success': False, 'error': str(e)}
        print(f"OrthoRoute engine import failed: {e}")
    
    # Test API bridge import
    try:
        from api_bridge import get_api_bridge
        bridge = get_api_bridge()
        api_info = bridge.get_api_info()
        import_results['api_bridge'] = {
            'success': True, 
            'api_info': api_info
        }
        print(f"API bridge imported successfully (using {api_info['current_api']} API)")
    except ImportError as e:
        import_results['api_bridge'] = {'success': False, 'error': str(e)}
        print(f"API bridge import failed: {e}")
    
    # Test IPC API availability
    try:
        from kicad.pcbnew.board import Board as IPCBoard
        import_results['ipc_api'] = {'success': True}
        print("IPC API (kicad-python) available")
    except ImportError as e:
        import_results['ipc_api'] = {'success': False, 'error': str(e)}
        print(f"IPC API not available: {e}")
    
    return import_results

def test_kicad_api():
    """Test KiCad API functionality"""
    api_results = {}
    
    try:
        import pcbnew
        
        # Test board creation
        try:
            board = pcbnew.BOARD()
            api_results['board_creation'] = {'success': True}
            print("✅ Board creation successful")
        except Exception as e:
            api_results['board_creation'] = {'success': False, 'error': str(e)}
            print(f"❌ Board creation failed: {e}")
            return api_results
        
        # Test basic board operations
        try:
            # Set board bounds
            bounds = board.GetBoardEdgesBoundingBox()
            layers = board.GetCopperLayerCount()
            api_results['board_operations'] = {
                'success': True,
                'layers': layers,
                'bounds_available': bounds is not None
            }
            print(f"✅ Board operations successful (layers: {layers})")
        except Exception as e:
            api_results['board_operations'] = {'success': False, 'error': str(e)}
            print(f"❌ Board operations failed: {e}")
        
        # Test net creation
        try:
            net_info = pcbnew.NETINFO_ITEM(board, "TEST_NET")
            board.Add(net_info)
            api_results['net_creation'] = {'success': True}
            print("✅ Net creation successful")
        except Exception as e:
            api_results['net_creation'] = {'success': False, 'error': str(e)}
            print(f"❌ Net creation failed: {e}")
        
        # Test footprint creation
        try:
            footprint = pcbnew.FOOTPRINT(board)
            footprint.SetReference("TEST_FP")
            board.Add(footprint)
            api_results['footprint_creation'] = {'success': True}
            print("✅ Footprint creation successful")
        except Exception as e:
            api_results['footprint_creation'] = {'success': False, 'error': str(e)}
            print(f"❌ Footprint creation failed: {e}")
        
        # Test track creation
        try:
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(pcbnew.VECTOR2I(0, 0))
            track.SetEnd(pcbnew.VECTOR2I(1000000, 1000000))  # 1mm in nanometers
            track.SetLayer(pcbnew.F_Cu)
            track.SetWidth(200000)  # 0.2mm
            board.Add(track)
            api_results['track_creation'] = {'success': True}
            print("✅ Track creation successful")
        except Exception as e:
            api_results['track_creation'] = {'success': False, 'error': str(e)}
            print(f"❌ Track creation failed: {e}")
    
    except ImportError as e:
        api_results['critical_error'] = f"pcbnew not available: {e}"
        print(f"❌ CRITICAL: pcbnew not available: {e}")
    
    return api_results

def test_board_operations():
    """Test board-specific operations"""
    board_results = {}
    
    try:
        import pcbnew
        
        # Try to get current board (might not exist in headless mode)
        try:
            board = pcbnew.GetBoard()
            if board:
                board_results['board_loaded'] = {'success': True}
                print("✅ Current board available")
                
                # Test board analysis
                try:
                    bounds = board.GetBoardEdgesBoundingBox()
                    width = bounds.GetWidth()
                    height = bounds.GetHeight()
                    layers = board.GetCopperLayerCount()
                    
                    footprints = list(board.GetFootprints())
                    netcodes = board.GetNetsByNetcode()
                    
                    board_results['board_analysis'] = {
                        'success': True,
                        'width_mm': width / 1e6,
                        'height_mm': height / 1e6,
                        'layers': layers,
                        'footprint_count': len(footprints),
                        'net_count': len(netcodes)
                    }
                    
                    print(f"✅ Board analysis: {width/1e6:.1f}x{height/1e6:.1f}mm, "
                          f"{layers} layers, {len(footprints)} footprints, {len(netcodes)} nets")
                    
                except Exception as e:
                    board_results['board_analysis'] = {'success': False, 'error': str(e)}
                    print(f"❌ Board analysis failed: {e}")
            else:
                board_results['board_loaded'] = {'success': False, 'reason': 'No board available'}
                print("⚠️ No current board available (expected in headless mode)")
        
        except Exception as e:
            board_results['board_loaded'] = {'success': False, 'error': str(e)}
            print(f"⚠️ Board loading failed: {e}")
        
        # Test synthetic board creation for routing tests
        try:
            test_board = create_test_board()
            if test_board:
                board_results['synthetic_board'] = {'success': True}
                print("✅ Synthetic test board created successfully")
            else:
                board_results['synthetic_board'] = {'success': False, 'reason': 'Creation returned None'}
                print("❌ Synthetic board creation returned None")
        except Exception as e:
            board_results['synthetic_board'] = {'success': False, 'error': str(e)}
            print(f"❌ Synthetic board creation failed: {e}")
    
    except ImportError as e:
        board_results['critical_error'] = f"pcbnew not available: {e}"
        print(f"❌ CRITICAL: pcbnew not available: {e}")
    
    return board_results

def create_test_board():
    """Create a synthetic test board for routing tests"""
    try:
        import pcbnew
        
        # Create board
        board = pcbnew.BOARD()
        
        # Create test nets
        net1 = pcbnew.NETINFO_ITEM(board, "NET_A")
        net2 = pcbnew.NETINFO_ITEM(board, "NET_B")
        board.Add(net1)
        board.Add(net2)
        
        # Create test footprints with pads
        fp1 = pcbnew.FOOTPRINT(board)
        fp1.SetReference("U1")
        fp1.SetPosition(pcbnew.VECTOR2I(5000000, 5000000))  # 5mm, 5mm
        
        fp2 = pcbnew.FOOTPRINT(board)
        fp2.SetReference("U2")
        fp2.SetPosition(pcbnew.VECTOR2I(15000000, 15000000))  # 15mm, 15mm
        
        # Add pads to footprints
        pad1 = pcbnew.PAD(fp1)
        pad1.SetName("1")
        pad1.SetPosition(pcbnew.VECTOR2I(5000000, 5000000))
        pad1.SetSize(pcbnew.VECTOR2I(1000000, 1000000))  # 1mm x 1mm
        pad1.SetNet(net1)
        fp1.Add(pad1)
        
        pad2 = pcbnew.PAD(fp1)
        pad2.SetName("2")
        pad2.SetPosition(pcbnew.VECTOR2I(6000000, 5000000))
        pad2.SetSize(pcbnew.VECTOR2I(1000000, 1000000))
        pad2.SetNet(net2)
        fp1.Add(pad2)
        
        pad3 = pcbnew.PAD(fp2)
        pad3.SetName("1")
        pad3.SetPosition(pcbnew.VECTOR2I(15000000, 15000000))
        pad3.SetSize(pcbnew.VECTOR2I(1000000, 1000000))
        pad3.SetNet(net1)
        fp2.Add(pad3)
        
        pad4 = pcbnew.PAD(fp2)
        pad4.SetName("2")
        pad4.SetPosition(pcbnew.VECTOR2I(16000000, 15000000))
        pad4.SetSize(pcbnew.VECTOR2I(1000000, 1000000))
        pad4.SetNet(net2)
        fp2.Add(pad4)
        
        # Add footprints to board
        board.Add(fp1)
        board.Add(fp2)
        
        print("📋 Created test board with 2 footprints, 4 pads, 2 nets")
        return board
        
    except Exception as e:
        print(f"❌ Test board creation failed: {e}")
        return None

def test_routing_engine():
    """Test OrthoRoute routing engine"""
    routing_results = {}
    
    try:
        from orthoroute_engine import OrthoRouteEngine
        
        # Test engine creation
        try:
            engine = OrthoRouteEngine()
            routing_results['engine_creation'] = {'success': True}
            print("✅ Routing engine created successfully")
        except Exception as e:
            routing_results['engine_creation'] = {'success': False, 'error': str(e)}
            print(f"❌ Routing engine creation failed: {e}")
            return routing_results
        
        # Test with synthetic board data
        try:
            test_board_data = {
                'bounds': {
                    'width_nm': 20000000,  # 20mm
                    'height_nm': 20000000,  # 20mm
                    'layers': 2
                },
                'nets': [
                    {
                        'id': 1,
                        'name': 'NET_A',
                        'pins': [
                            {'x': 5000000, 'y': 5000000, 'layer': 0},
                            {'x': 15000000, 'y': 15000000, 'layer': 0}
                        ],
                        'width_nm': 200000,
                        'kicad_net': None
                    },
                    {
                        'id': 2,
                        'name': 'NET_B',
                        'pins': [
                            {'x': 6000000, 'y': 5000000, 'layer': 0},
                            {'x': 16000000, 'y': 15000000, 'layer': 0}
                        ],
                        'width_nm': 200000,
                        'kicad_net': None
                    }
                ],
                'obstacles': {}
            }
            
            config = {
                'grid_pitch_mm': 0.2,
                'max_iterations': 2,
                'via_cost': 10,
                'batch_size': 5,
                'debug_output': True,
                'show_progress': False
            }
            
            # Test board data loading
            load_success = engine.load_board_data(test_board_data)
            if load_success:
                routing_results['board_data_loading'] = {'success': True}
                print("✅ Board data loading successful")
            else:
                routing_results['board_data_loading'] = {'success': False, 'reason': 'load_board_data returned False'}
                print("❌ Board data loading failed")
                return routing_results
            
            # Test routing (without actual board for track creation)
            print("🔄 Testing routing algorithm...")
            results = engine.route(test_board_data, config, board=None)
            
            if results and results.get('success'):
                routing_results['routing_execution'] = {
                    'success': True,
                    'nets_processed': results['stats']['total_nets'],
                    'successful_nets': results['stats']['successful_nets'],
                    'success_rate': results['stats']['success_rate']
                }
                print(f"✅ Routing successful: {results['stats']['successful_nets']}/{results['stats']['total_nets']} nets")
            else:
                error = results.get('error', 'Unknown error') if results else 'No results returned'
                routing_results['routing_execution'] = {'success': False, 'error': error}
                print(f"❌ Routing failed: {error}")
        
        except Exception as e:
            routing_results['routing_test'] = {'success': False, 'error': str(e)}
            print(f"❌ Routing test failed: {e}")
            traceback.print_exc()
    
    except ImportError as e:
        routing_results['critical_error'] = f"OrthoRoute engine not available: {e}"
        print(f"❌ CRITICAL: OrthoRoute engine not available: {e}")
    
    return routing_results

def assess_overall_success(test_results):
    """Assess overall success of test suite"""
    critical_failures = []
    
    # Check for critical import failures
    if not test_results['import_tests'].get('pcbnew', {}).get('success', False):
        critical_failures.append("pcbnew import failed")
    
    if not test_results['import_tests'].get('orthoroute_engine', {}).get('success', False):
        critical_failures.append("OrthoRoute engine import failed")
    
    # Check for critical API failures
    if not test_results['api_tests'].get('board_creation', {}).get('success', False):
        critical_failures.append("Board creation failed")
    
    # Check for routing engine failures
    if not test_results['routing_tests'].get('engine_creation', {}).get('success', False):
        critical_failures.append("Routing engine creation failed")
    
    if critical_failures:
        print(f"\n❌ CRITICAL FAILURES: {', '.join(critical_failures)}")
        return False
    else:
        print("\n✅ ALL CRITICAL TESTS PASSED")
        return True

def print_test_summary(test_results):
    """Print comprehensive test summary"""
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    # Import tests
    print("\n🔧 Import Tests:")
    for test_name, result in test_results['import_tests'].items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {test_name}")
    
    # API tests
    print("\n🔌 API Tests:")
    for test_name, result in test_results['api_tests'].items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {test_name}")
    
    # Board tests
    print("\n📋 Board Tests:")
    for test_name, result in test_results['board_tests'].items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {test_name}")
    
    # Routing tests
    print("\n🚀 Routing Tests:")
    for test_name, result in test_results['routing_tests'].items():
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {test_name}")
    
    # Overall result
    overall_status = "✅ PASS" if test_results['overall_success'] else "❌ FAIL"
    print(f"\n🎯 OVERALL RESULT: {overall_status}")

def save_test_results(test_results):
    """Save test results to JSON file"""
    try:
        results_file = current_dir / "headless_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\n💾 Test results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Could not save test results: {e}")

if __name__ == "__main__":
    # Run headless tests
    results = test_orthoroute_headless()
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_success'] else 1
    print(f"\n🚪 Exiting with code: {exit_code}")
    sys.exit(exit_code)
