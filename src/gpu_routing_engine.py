#!/usr/bin/env python3
"""
GPU Routing Engine - Revolutionary CUDA-accelerated autorouter
Integrates with reverse-engineered KiCad IPC APIs for professional routing
"""

import logging
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class RoutingResult:
    """Result of GPU routing operation"""
    success: bool
    nets_routed: int = 0
    tracks_created: int = 0
    vias_created: int = 0
    success_rate: float = 0.0
    routing_time: float = 0.0
    gpu_time: float = 0.0
    error: Optional[str] = None
    failed_nets: List[str] = None
    detailed_stats: Dict[str, Any] = None

class RevolutionaryGPUEngine:
    """
    Revolutionary GPU routing engine using undocumented IPC APIs
    Combines CUDA acceleration with direct C++ connectivity access
    """
    
    def __init__(self, connectivity_data: Dict[str, Any]):
        self.connectivity_data = connectivity_data
        self.gpu_available = self._check_gpu_availability()
        
        # Optimized settings for revolutionary performance
        self.settings = {
            'grid_resolution': 0.05,    # 0.05mm grid for high precision
            'via_cost': 30,             # Balanced via usage
            'max_iterations': 500,      # High iteration count for quality
            'memory_efficiency': True,  # Optimize GPU memory usage
            'parallel_nets': 8,         # Route multiple nets simultaneously
            'real_time_feedback': True, # Use IPC APIs for live updates
            'gpu_memory_limit': 0.85,   # Use 85% of available GPU memory
            'wavefront_batch_size': 10000,  # Large batches for GPU efficiency
            'cuda_optimization': True   # Enable all CUDA optimizations
        }
        
        logger.info(f"Revolutionary GPU Engine initialized")
        logger.info(f"GPU Available: {self.gpu_available}")
        logger.info(f"Connectivity data: {len(connectivity_data.get('net_details', []))} nets")
    
    def _check_gpu_availability(self) -> bool:
        """Check if CUDA-capable GPU is available"""
        try:
            import cupy as cp
            
            # Test basic GPU functionality
            test_array = cp.array([1, 2, 3, 4, 5])
            result = cp.sum(test_array)
            
            gpu_info = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = gpu_info['name'].decode()
            gpu_memory = gpu_info['totalGlobalMem'] / (1024**3)  # GB
            
            logger.info(f"✓ GPU detected: {gpu_name}")
            logger.info(f"✓ GPU memory: {gpu_memory:.1f} GB")
            logger.info(f"✓ CUDA cores: {gpu_info['multiProcessorCount'] * 128}")  # Approximate
            
            return True
            
        except ImportError:
            logger.warning("CuPy not available - GPU acceleration disabled")
            return False
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
        
    def update_settings(self, settings: Dict):
        """Update routing settings"""
        self.settings.update(settings)
        logger.info(f"Updated routing settings: {settings}")
    
    def route_all_nets(self, progress_callback=None) -> RoutingResult:
        """Route all unrouted nets using GPU acceleration"""
        logger.info("🚀 Starting GPU routing of all nets...")
        start_time = time.time()
        
        try:
            # Setup temporary directory for communication
            self.temp_dir = tempfile.mkdtemp(prefix="orthoroute_")
            logger.info(f"Created temp directory: {self.temp_dir}")
            
            # Prepare routing request
            request_data = self._prepare_routing_request()
            
            # Check if there are any nets to route
            nets_to_route = len(request_data.get('nets', []))
            if nets_to_route == 0:
                # Enhanced debugging for plane-aware routing
                all_nets = self.board_data.get('all_nets_debug', [])
                copper_zones = self.board_data.get('copper_zones', [])
                plane_nets = set(zone['net'] for zone in copper_zones if zone.get('filled', True))
                
                logger.warning("⚠️  NO NETS TO ROUTE - Enhanced Analysis:")
                logger.warning(f"    📋 Total nets in board: {len(all_nets)}")
                logger.warning(f"    ⚡ Plane-connected nets: {len(plane_nets)} - {sorted(plane_nets) if plane_nets else 'None'}")
                logger.warning(f"    🔌 Remaining routable nets: {nets_to_route}")
                logger.warning("⚠️  Possible causes:")
                logger.warning("    1. All nets are connected via copper planes (GND, power)")
                logger.warning("    2. Board has no point-to-point connections")
                logger.warning("    3. All nets already routed with tracks") 
                logger.warning("    4. KiCad board data extraction failed")
                return RoutingResult(
                    success=False,
                    error=f"No nets to route (found {len(plane_nets)} plane-connected nets)",
                    routing_time=time.time() - start_time,
                    tracks_created=0,
                    vias_created=0,
                    success_rate=0.0
                )
            
            request_file = os.path.join(self.temp_dir, "routing_request.json")
            
            with open(request_file, 'w') as f:
                json.dump(request_data, f, indent=2)
            
            # Launch GPU routing server
            result = self._launch_gpu_server(progress_callback)
            
            # Apply results to KiCad
            if result.success:
                self._apply_routing_results()
            
            return result
            
        except Exception as e:
            logger.error(f"❌ GPU routing failed: {e}")
            return RoutingResult(
                success=False,
                error=str(e),
                routing_time=time.time() - start_time
            )
        finally:
            self._cleanup()
    
    def _prepare_routing_request(self) -> Dict:
        """Prepare routing request data"""
        # Extract unrouted nets
        all_nets = self.board_data.get('nets', [])
        unrouted_nets = [net for net in all_nets 
                        if not net.get('routed', False)]
        
        # Debug logging for net analysis - MOVED TO EARLY POSITION
        logger.info(f"[DEBUG] Total nets in board_data: {len(all_nets)}")
        logger.info(f"[DEBUG] Unrouted nets found: {len(unrouted_nets)}")
        
        # Save debug info to file IMMEDIATELY - before any routing attempts
        self._save_debug_log(all_nets, unrouted_nets)
        
        if len(all_nets) == 0:
            logger.warning("⚠️  No nets found in board_data! This may be why routing completes instantly.")
        elif len(unrouted_nets) == 0:
            logger.warning("⚠️  All nets are already routed! This may be why routing completes instantly.")
        else:
            for i, net in enumerate(unrouted_nets[:3]):  # Show first 3 nets
                logger.info(f"[DEBUG] Net {i+1}: {net.get('name', 'Unknown')} - {len(net.get('pins', []))} pins")
        
        # Get board bounds with enhanced debugging
        bounds = self.board_data.get('bounds', (0, 0, 100, 80))
        logger.info(f"[REQUEST DEBUG] Board bounds from board_data: {bounds}")
        
        request = {
            'board_info': {
                'width_mm': bounds[2] - bounds[0],
                'height_mm': bounds[3] - bounds[1],
                'layers': self.board_data.get('layers', 2),
                'bounds': bounds
            },
            'nets': unrouted_nets,
            'components': self.board_data.get('components', []),
            'existing_tracks': self.board_data.get('tracks', []),
            'settings': self.settings,
            'temp_dir': self.temp_dir
        }
        
        logger.info(f"[REQUEST DEBUG] board_info bounds: {request['board_info']['bounds']}")
        logger.info(f"[REQUEST DEBUG] Full request structure: {list(request.keys())}")
        logger.info(f"[REQUEST DEBUG] board_info structure: {list(request['board_info'].keys())}")
        logger.info(f"Prepared routing request: {len(unrouted_nets)} nets to route")
        return request
    
    def _save_debug_log(self, all_nets, unrouted_nets):
        """Save debug information to file - called early in process"""
        try:
            # Save to Documents folder on Windows
            import os
            if os.name == 'nt':  # Windows
                documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
                debug_file = os.path.join(documents_path, "orthoroute_debug.log")
            else:
                debug_file = os.path.join(os.path.expanduser('~'), "orthoroute_debug.log")
            
            with open(debug_file, 'w') as f:
                f.write(f"OrthoRoute Debug Log\n")
                f.write(f"===================\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total nets in board_data: {len(all_nets)}\n")
                f.write(f"Unrouted nets found: {len(unrouted_nets)}\n")
                f.write(f"Board filename: {self.board_data.get('filename', 'Unknown')}\n")
                f.write(f"Board dimensions: {self.board_data.get('width', 0):.1f} x {self.board_data.get('height', 0):.1f} mm\n")
                f.write(f"Board bounds: {self.board_data.get('bounds', 'Unknown')}\n")
                f.write(f"Layers: {self.board_data.get('layers', 0)}\n")
                f.write(f"Components: {len(self.board_data.get('components', []))}\n")
                f.write(f"Pads: {len(self.board_data.get('pads', []))}\n")
                f.write(f"Existing tracks: {len(self.board_data.get('tracks', []))}\n")
                f.write(f"Existing vias: {len(self.board_data.get('vias', []))}\n")
                f.write(f"\n")
                f.write(f"Debug file saved to: {debug_file}\n")
                f.write(f"\n")
                if len(all_nets) > 0:
                    f.write(f"All nets details:\n")
                    for i, net in enumerate(all_nets):
                        pins = len(net.get('pins', []))
                        routed = net.get('routed', False)
                        pin_list = net.get('pins', [])
                        f.write(f"  {i+1}. '{net.get('name', 'Unknown')}' - {pins} pins - {'ROUTED' if routed else 'UNROUTED'}\n")
                        if pins > 0:
                            for j, pin in enumerate(pin_list[:3]):  # Show first 3 pins
                                f.write(f"      Pin {j+1}: x={pin.get('x', 0):.3f}, y={pin.get('y', 0):.3f}\n")
                        f.write(f"\n")
                else:
                    f.write("❌ NO NETS FOUND - This explains why routing completes instantly!\n")
                    f.write("Possible causes:\n")
                    f.write("  1. Board has no electrical connections defined\n")
                    f.write("  2. KiCad IPC API connection failed\n")
                    f.write("  3. Board data extraction failed\n")
                    f.write("  4. Plugin is running on wrong board/no board loaded\n")
                f.write(f"\n")
            
            logger.info(f"🐛 Debug log saved to: {debug_file}")
            
        except Exception as e:
            logger.warning(f"Could not write debug file: {e}")
            # Try alternative location
            try:
                fallback_file = os.path.join(os.getcwd(), "orthoroute_debug.log")
                with open(fallback_file, 'w') as f:
                    f.write(f"Debug fallback: Total nets={len(all_nets)}, Unrouted={len(unrouted_nets)}\n")
                logger.info(f"🐛 Fallback debug log saved to: {fallback_file}")
            except:
                pass
    
    def _launch_gpu_server(self, progress_callback=None) -> RoutingResult:
        """Launch GPU routing server in separate process"""
        try:
            # Create GPU routing server script
            server_script = self._create_gpu_server_script()
            
            # Launch server process
            logger.info("Launching GPU routing server...")
            self.routing_process = subprocess.Popen(
                [sys.executable, server_script],
                cwd=self.temp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor progress
            return self._monitor_routing_progress(progress_callback)
            
        except Exception as e:
            logger.error(f"Failed to launch GPU server: {e}")
            return RoutingResult(success=False, error=str(e))
    
    def _create_gpu_server_script(self) -> str:
        """Create GPU routing server script"""
        server_script = os.path.join(self.temp_dir, "gpu_routing_server.py")
        
        script_content = '''#!/usr/bin/env python3
"""
GPU Routing Server - Isolated GPU routing process
"""

import sys
import json
import time
import logging
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main GPU routing server"""
    try:
        # Load routing request
        with open('routing_request.json', 'r') as f:
            request = json.load(f)
        
        logger.info("🔥 GPU Routing Server started")
        logger.info(f"Board: {request['board_info']['width_mm']:.1f}x{request['board_info']['height_mm']:.1f}mm")
        logger.info(f"Nets to route: {len(request['nets'])}")
        
        # Check GPU availability
        gpu_available = check_gpu()
        
        # Run routing
        result = run_routing(request, gpu_available)
        
        # Save results
        with open('routing_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✅ Routing completed: {result['success_rate']:.1f}% success")
        return 0
        
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        traceback.print_exc()
        
        # Save error result
        error_result = {
            'success': False,
            'error': str(e),
            'tracks_created': 0,
            'success_rate': 0.0
        }
        
        try:
            with open('routing_result.json', 'w') as f:
                json.dump(error_result, f)
        except:
            pass
        
        return 1

def check_gpu():
    """Check GPU availability"""
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props["name"].decode("utf-8")
        logger.info(f"✅ GPU available: {gpu_name}")
        return True
    except Exception as e:
        logger.info(f"❌ GPU not available: {e}")
        return False

# Import the real routing function from the end of this file
def run_routing(request, gpu_available):
    """Call the real routing function"""
    return run_routing_real(request, gpu_available)

def route_net_gpu(net, settings):
    """Route a single net using GPU"""
    try:
        import cupy as cp
        # Mock GPU routing algorithm
        grid_pitch = settings.get('grid_pitch', 0.1)
        max_iter = settings.get('max_iterations', 200)  # Increased default for mock routing too
        
        # Simulate GPU pathfinding
        time.sleep(0.05)  # Simulate GPU computation
        
        # 85% success rate for demo
        import random
        return random.random() < 0.85
        
    except Exception as e:
        logger.error(f"GPU routing failed: {e}")
        return False

def route_net_cpu(net, settings):
    """Route a single net using CPU fallback"""
    try:
        # Mock CPU routing algorithm
        time.sleep(0.1)  # Simulate CPU computation
        
        # 70% success rate for CPU mode
        import random
        return random.random() < 0.70
        
    except Exception as e:
        logger.error(f"CPU routing failed: {e}")
        return False

if __name__ == "__main__":
    import sys
    sys.exit(main())
'''
        
        with open(server_script, 'w') as f:
            f.write(script_content)
        
        return server_script
    
    def _monitor_routing_progress(self, progress_callback=None) -> RoutingResult:
        """Monitor routing progress and return result with live updates"""
        status_file = os.path.join(self.temp_dir, "routing_status.json")
        result_file = os.path.join(self.temp_dir, "routing_result.json")
        
        try:
            # Monitor progress with enhanced live data
            while self.routing_process and self.routing_process.poll() is None:
                if os.path.exists(status_file):
                    try:
                        with open(status_file, 'r') as f:
                            status = json.load(f)
                        
                        if progress_callback:
                            # Enhanced progress callback with all live data
                            progress_callback(
                                status.get('progress', 0),
                                status.get('current_net', 'Unknown'),
                                status.get('status', 'running'),
                                {
                                    'nets_completed': status.get('nets_completed', 0),
                                    'total_nets': status.get('total_nets', 0),
                                    'success_rate': status.get('success_rate', 0.0),
                                    'tracks_created': status.get('tracks_created', 0),
                                    'vias_created': status.get('vias_created', 0),
                                    'elapsed_time': status.get('elapsed_time', 0.0),
                                    'live_tracks': status.get('live_tracks', []),
                                    'timestamp': status.get('timestamp', time.time())
                                }
                            )
                    except:
                        pass
                
                time.sleep(0.1)  # Check every 100ms for more responsive animation
            
            # Get final result
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                return RoutingResult(
                    success=result_data.get('success', False),
                    tracks_created=result_data.get('tracks_created', 0),
                    vias_created=result_data.get('vias_created', 0),
                    success_rate=result_data.get('success_rate', 0.0),
                    routing_time=time.time() - time.time(),  # Will be updated
                    error=result_data.get('error')
                )
            else:
                return RoutingResult(success=False, error="No result file generated")
                
        except Exception as e:
            logger.error(f"Error monitoring progress: {e}")
            return RoutingResult(success=False, error=str(e))
    
    def _apply_routing_results(self):
        """Apply routing results to KiCad"""
        result_file = os.path.join(self.temp_dir, "routing_result.json")
        
        try:
            with open(result_file, 'r') as f:
                result = json.load(f)
            
            tracks = result.get('tracks', [])
            vias = result.get('vias', [])
            
            logger.info(f"Applying {len(tracks)} tracks and {len(vias)} vias to KiCad")
            
            # Apply tracks
            for track in tracks:
                self.kicad_interface.create_track(
                    track['start']['x'],
                    track['start']['y'],
                    track['end']['x'],
                    track['end']['y'],
                    track.get('layer', 'F.Cu'),
                    track.get('width', 0.2),
                    track.get('net_name', '')
                )
            
            # Apply vias
            for via in vias:
                self.kicad_interface.create_via(
                    via['x'], via['y'],
                    via.get('size', 0.4),
                    via.get('drill', 0.2),
                    via.get('from_layer', 'F.Cu'),
                    via.get('to_layer', 'B.Cu'),
                    via.get('net_name', '')
                )
            
            # Refresh KiCad
            self.kicad_interface.refresh_board()
            
        except Exception as e:
            logger.error(f"Error applying results: {e}")
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info("Cleaned up temporary directory")
            except Exception as e:
                logger.error(f"Error cleaning up: {e}")
    
    def stop_routing(self):
        """Stop the routing process"""
        if self.routing_process:
            try:
                self.routing_process.terminate()
                self.routing_process.wait(timeout=5)
                logger.info("Routing process stopped")
            except:
                self.routing_process.kill()
                logger.info("Routing process killed")
            finally:
                self.routing_process = None

def run_routing_real(request, gpu_available):
    """Run the actual routing with real pathfinding"""
    nets = request['nets']
    settings = request['settings']
    board_info = request.get('board_info', {})  # FIXED: use 'board_info' instead of 'board'
    existing_tracks = request.get('existing_tracks', [])
    components = request.get('components', [])
    
    logger.info(f"[*] Starting routing process")
    logger.info(f"[#] Input data: {len(nets)} nets, {len(existing_tracks)} tracks, {len(components)} components")
    logger.info(f"[+] GPU available: {gpu_available}")
    logger.info(f"[=] Board info: {board_info}")
    logger.info(f"[>] Settings: {settings}")
    
    # Initialize progress tracking with empty live tracks
    try:
        with open('routing_status.json', 'w') as f:
            json.dump({
                'progress': 0, 
                'current_net': 'Initializing...', 
                'status': 'running',
                'live_tracks': [],
                'all_tracks': [],
                'timestamp': time.time()
            }, f)
    except:
        pass  # File creation might fail in some environments
    
    # Validate input data
    if not nets:
        logger.warning("[X] No nets to route!")
        return {
            'success': False,
            'tracks_created': 0,
            'vias_created': 0,
            'success_rate': 0,
            'error': 'No nets to route'
        }
    
    # Log first few nets for debugging
    logger.info(f"[-] First few nets:")
    for i, net in enumerate(nets[:3]):
        pins = net.get('pins', [])
        logger.info(f"  Net {i+1}: {net.get('name', 'Unknown')} - {len(pins)} pins")
        if pins:
            logger.info(f"    Pin 1: ({pins[0].get('x', 0):.3f}, {pins[0].get('y', 0):.3f})")
            if len(pins) > 1:
                logger.info(f"    Pin 2: ({pins[1].get('x', 0):.3f}, {pins[1].get('y', 0):.3f})")
    
    # Prepare routing environment with enhanced bounds debugging
    bounds_raw = board_info.get('bounds')
    logger.info(f"[BOUNDS DEBUG] Raw bounds from board_info: {bounds_raw} (type: {type(bounds_raw)})")
    
    if bounds_raw is None:
        logger.warning("[BOUNDS DEBUG] bounds_raw is None, using hardcoded fallback!")
        bounds = {'min_x': -50, 'max_x': 50, 'min_y': -50, 'max_y': 50}
    elif isinstance(bounds_raw, (list, tuple)) and len(bounds_raw) == 4:
        bounds = {'min_x': bounds_raw[0], 'min_y': bounds_raw[1], 'max_x': bounds_raw[2], 'max_y': bounds_raw[3]}
        logger.info(f"[BOUNDS DEBUG] Converted tuple bounds: {bounds}")
    elif isinstance(bounds_raw, dict):
        bounds = bounds_raw
        logger.info(f"[BOUNDS DEBUG] Using dict bounds directly: {bounds}")
    else:
        logger.warning(f"[BOUNDS DEBUG] Unexpected bounds format: {bounds_raw}, using fallback")
        bounds = {'min_x': -50, 'max_x': 50, 'min_y': -50, 'max_y': 50}
    
    layers_raw = board_info.get('layers', 2)
    # Defensive layer count parsing - handle both integers and layer arrays
    if isinstance(layers_raw, (list, tuple)):
        layers = len(layers_raw) if layers_raw else 2
    elif isinstance(layers_raw, (int, float)):
        layers = int(layers_raw)
    else:
        layers = 2
    
    # Ensure minimum layer count
    layers = max(layers, 1)
    grid_pitch = settings.get('grid_pitch', 0.1)
    
    logger.info(f"[=] Routing bounds: {bounds}")
    logger.info(f"[:] Layer count: {layers} (from {type(layers_raw).__name__}: {layers_raw}), Grid pitch: {grid_pitch}mm")
    
    # Build obstacle map from existing tracks and components
    obstacles = []
    
    # Add existing tracks as obstacles
    for track in existing_tracks:
        start = track.get('start', {})
        end = track.get('end', {})
        obstacles.append({
            'x': start.get('x', 0),
            'y': start.get('y', 0),
            'layer': 0 if track.get('layer', 'F.Cu') == 'F.Cu' else 1
        })
        obstacles.append({
            'x': end.get('x', 0),
            'y': end.get('y', 0),
            'layer': 0 if track.get('layer', 'F.Cu') == 'F.Cu' else 1
        })
    
    # Add component pads as obstacles for other nets
    for component in components:
        pads = component.get('pads', [])
        for pad in pads:
            obstacles.append({
                'x': pad.get('x', 0),
                'y': pad.get('y', 0),
                'layer': 0  # Assume front layer for now
            })
    
    logger.info(f"[!] Total obstacles: {len(obstacles)}")
    
    # Update settings with routing environment
    routing_settings = settings.copy()
    routing_settings.update({
        'obstacles': obstacles,
        'bounds': bounds,
        'layers': layers
    })
    
    routed_tracks = []
    routed_vias = []
    successful_nets = 0
    
    # Route each net
    for i, net in enumerate(nets):
        # Update progress while preserving existing tracks
        progress = int((i / len(nets)) * 100) if nets else 100
        try:
            # Read existing status to preserve live tracks
            existing_status = {}
            try:
                with open('routing_status.json', 'r') as f:
                    existing_status = json.load(f)
            except:
                pass
                
            with open('routing_status.json', 'w') as f:
                json.dump({
                    'progress': progress,
                    'current_net': net.get('name', f'Net_{i}'),
                    'status': 'routing',
                    'live_tracks': existing_status.get('live_tracks', []),
                    'all_tracks': existing_status.get('all_tracks', []),
                    'timestamp': time.time()
                }, f)
        except:
            pass
        
        net_name = net.get('name', f'Net_{i}')
        pins = net.get('pins', [])
        
        logger.info(f"[o] Routing net {i+1}/{len(nets)}: {net_name} ({len(pins)} pins)")
        
        # Skip nets with insufficient pins
        if len(pins) < 2:
            logger.warning(f"[!] Skipping {net_name} - insufficient pins ({len(pins)})")
            continue
        
        # Route the net using GPU or CPU
        if gpu_available:
            from routing_algorithms import route_net_gpu_streaming
            success, tracks = route_net_gpu_streaming(net, routing_settings)
            
            if success and tracks:
                # GPU streaming already generated tracks - use them directly!
                routed_tracks.extend(tracks)
                logger.info(f"[GPU] Streamed {len(tracks)} track segments for {net_name}")
            elif success:
                # Path found but no tracks - generate simple track
                from routing_algorithms import generate_track_geometry
                tracks, vias = generate_track_geometry(net, routing_settings, bounds, grid_pitch)
                routed_tracks.extend(tracks)
                routed_vias.extend(vias)
                logger.info(f"[GPU] Generated {len(tracks)} tracks for {net_name}")
        else:
            from routing_algorithms import route_net_cpu
            success = route_net_cpu(net, routing_settings)
            
            if success:
                # CPU routing needs track generation
                from routing_algorithms import generate_track_geometry
                tracks, vias = generate_track_geometry(net, routing_settings, bounds, grid_pitch)
                routed_tracks.extend(tracks)
                routed_vias.extend(vias)
                logger.info(f"[CPU] Generated {len(tracks)} tracks and {len(vias)} vias for {net_name}")
        
        if success:
            successful_nets += 1
            logger.info(f"[OK] Successfully routed {net_name}")
            
            # Update live status with new tracks for real-time visualization
            try:
                current_progress = ((i + 1) / len(nets)) * 100
                status_data = {
                    'progress': current_progress,
                    'current_net': net_name,
                    'status': 'routing',
                    'nets_completed': successful_nets,
                    'total_nets': len(nets),
                    'tracks_created': len(routed_tracks),
                    'vias_created': len(routed_vias),
                    'success_rate': (successful_nets / (i + 1)) * 100,
                    'live_tracks': routed_tracks[-len(tracks):] if tracks else [],  # Show newest tracks as "live"
                    'all_tracks': routed_tracks,  # All tracks so far
                    'timestamp': time.time()
                }
                with open('routing_status.json', 'w') as f:
                    json.dump(status_data, f)
                logger.info(f"[STATUS] Wrote status with {len(status_data.get('live_tracks', []))} live tracks, {len(status_data.get('all_tracks', []))} total")
            except Exception as e:
                logger.warning(f"Failed to update live status: {e}")
            
            # Add new tracks as obstacles for subsequent nets
            for track in tracks:
                obstacles.append({
                    'x': track['start']['x'],
                    'y': track['start']['y'], 
                    'layer': 0 if track['layer'] == 'F.Cu' else 1
                })
                obstacles.append({
                    'x': track['end']['x'],
                    'y': track['end']['y'],
                    'layer': 0 if track['layer'] == 'F.Cu' else 1
                })
        else:
            logger.warning(f"[X] Failed to route {net_name}")
            
            # Update status even for failed routing
            try:
                current_progress = ((i + 1) / len(nets)) * 100
                with open('routing_status.json', 'w') as f:
                    json.dump({
                        'progress': current_progress,
                        'current_net': net_name,
                        'status': 'routing',
                        'nets_completed': successful_nets,
                        'total_nets': len(nets),
                        'tracks_created': len(routed_tracks),
                        'vias_created': len(routed_vias),
                        'success_rate': (successful_nets / (i + 1)) * 100,
                        'live_tracks': routed_tracks,  # Show all tracks for failed net (no new ones to highlight)  
                        'all_tracks': routed_tracks,  # All tracks so far
                        'timestamp': time.time()
                    }, f)
            except Exception as e:
                logger.warning(f"Failed to update live status: {e}")
    
    success_rate = (successful_nets / len(nets)) * 100 if nets else 0
    routed_vias = []    # No vias created yet
    successful_nets = 0 # No nets routed yet
    
    # Final status with complete track data for UI visualization
    try:
        with open('routing_status.json', 'w') as f:
            json.dump({
                'progress': 100, 
                'current_net': 'Complete', 
                'status': 'finished',
                'live_tracks': [],  # No more live tracks, routing complete
                'all_tracks': routed_tracks  # All completed tracks for final visualization
            }, f)
        logger.info(f"[STATUS] Wrote final status with {len(routed_tracks)} completed tracks")
    except Exception as e:
        logger.warning(f"Failed to write final status: {e}")
    
    logger.info(f"[END] Routing complete: {successful_nets}/{len(nets)} nets ({success_rate:.1f}%)")
    logger.info(f"[#] Final results: {len(routed_tracks)} tracks, {len(routed_vias)} vias")
    
    return {
        'success': True,
        'tracks_created': len(routed_tracks),
        'vias_created': len(routed_vias),
        'success_rate': success_rate,
        'routed_nets': successful_nets,
        'total_nets': len(nets),
        'tracks': routed_tracks,
        'vias': routed_vias
    }
