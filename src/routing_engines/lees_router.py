#!/usr/bin/env python3
"""
Lee's Algorithm Router Implementation

Implements Lee's wavefront expansion algorithm for PCB autorouting with GPU acceleration.
Extends the base router interface with Lee's specific pathfinding methodology.
"""
import logging
import time
import math
import sys
import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# Add src directory to Python path for direct execution
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try absolute imports first, fall back to relative
try:
    from routing_engines.base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from core.drc_rules import DRCRules
    from core.gpu_manager import GPUManager
    from core.board_interface import BoardInterface
    from data_structures.grid_config import GridConfig
except ImportError:
    from .base_router import BaseRouter, RoutingResult, RouteSegment, RoutingStats
    from ..core.drc_rules import DRCRules
    from ..core.gpu_manager import GPUManager
    from ..core.board_interface import BoardInterface
    from ..data_structures.grid_config import GridConfig

logger = logging.getLogger(__name__)

# GPU import handling
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


class LeeRouter(BaseRouter):
    """Lee's Algorithm Router with GPU acceleration and DRC awareness"""
    
    def __init__(self, board_interface: BoardInterface, drc_rules: DRCRules, 
                 gpu_manager: GPUManager, grid_config: GridConfig):
        """Initialize Lee's algorithm router"""
        super().__init__(board_interface, drc_rules, gpu_manager, grid_config)
        
        # Lee's algorithm specific configuration
        self.max_iterations = 10000  # Maximum wavefront expansion iterations
        # 8-connected neighbors for 45-degree routing (horizontal, vertical, and diagonal)
        self.neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),  # orthogonal
                                (-1, -1), (-1, 1), (1, -1), (1, 1)]   # diagonal
        
        # Pre-computed pad exclusion grids for performance optimization
        self.pad_exclusion_grids = {}  # Per-layer grids with ALL pad exclusions
        
        logger.info("🌊 Lee's Algorithm Router initialized")
        logger.info(f"   Max iterations: {self.max_iterations}")
        logger.info(f"   Connectivity: 8-connected neighbors (45° routing enabled)")
        
        # Pre-compute pad exclusions once at startup to eliminate O(N×P) bottleneck
        self._precompute_pad_exclusion_grids()
    
    def route_net(self, net_name: str, timeout: float = 10.0) -> RoutingResult:
        """Route a complete net using Lee's algorithm"""
        start_time = time.time()
        
        # Get pads for this net
        net_pads = self.board_interface.get_pads_for_net(net_name)
        if len(net_pads) < 2:
            logger.warning(f"⚠️ Net {net_name} has insufficient pads ({len(net_pads)})")
            return RoutingResult.FAILED
        
        logger.info(f"🔗 Routing net '{net_name}' with {len(net_pads)} pads using Lee's algorithm")
        
        # Get net-specific constraints
        net_constraints = self.drc_rules.get_net_constraints(net_name)
        
        try:
            if len(net_pads) == 2:
                # Simple two-pad connection
                result = self._route_two_pads_lee(
                    net_pads[0], net_pads[1], net_name, net_constraints, timeout, start_time
                )
            else:
                # Multi-pad net using minimum spanning tree approach
                result = self._route_multi_pad_net_lee(
                    net_pads, net_name, net_constraints, timeout, start_time
                )
            
            if result:
                # Update statistics
                self.stats.update_success(result)
                
                # Add to routed segments
                self.routed_segments.extend(result)
                
                # Notify callback if available
                if self.track_callback:
                    tracks = [seg for seg in result if seg.type == 'track']
                    vias = [seg for seg in result if seg.type == 'via']
                    self.track_callback(tracks, vias)
                
                logger.info(f"✅ Successfully routed {net_name}")
                return RoutingResult.SUCCESS
            else:
                logger.warning(f"❌ Failed to route {net_name}")
                return RoutingResult.FAILED
                
        except Exception as e:
            logger.error(f"❌ Exception routing {net_name}: {e}")
            return RoutingResult.FAILED
    
    def route_two_pads(self, pad_a: Dict, pad_b: Dict, net_name: str, 
                      timeout: float = 5.0) -> Optional[List[RouteSegment]]:
        """Route between two pads using Lee's algorithm"""
        start_time = time.time()
        net_constraints = self.drc_rules.get_net_constraints(net_name)
        
        return self._route_two_pads_lee(pad_a, pad_b, net_name, net_constraints, timeout, start_time)
    
    def _route_two_pads_lee(self, pad_a: Dict, pad_b: Dict, net_name: str, 
                           net_constraints: Dict, timeout: float, start_time: float) -> Optional[List[RouteSegment]]:
        """Route between two pads using Lee's wavefront expansion with multi-layer support"""
        
        # Calculate connection distance for routing strategy
        distance = self.board_interface.calculate_connection_distance(pad_a, pad_b)
        
        logger.info(f"🌊 Lee's routing {net_name}: distance={distance:.1f}mm")
        
        # STRATEGY 1: Try direct single-layer routing first (70% of timeout)
        single_layer_timeout = timeout * 0.7
        if time.time() - start_time < single_layer_timeout:
            
            # Try best layer first (usually F.Cu for surface mount components)
            best_layer = self._select_best_layer_for_connection(pad_a, pad_b)
            
            result = self._route_single_layer_lee(
                pad_a, pad_b, best_layer, net_name, net_constraints, 
                single_layer_timeout * 0.6, start_time
            )
            
            if result:
                logger.info(f"✅ Lee's direct routing succeeded on {best_layer}")
                return result
            
            # Try other layer
            other_layer = 'B.Cu' if best_layer == 'F.Cu' else 'F.Cu'
            result = self._route_single_layer_lee(
                pad_a, pad_b, other_layer, net_name, net_constraints,
                single_layer_timeout * 0.4, start_time
            )
            
            if result:
                logger.info(f"✅ Lee's direct routing succeeded on {other_layer}")
                return result
        
        # STRATEGY 2: Multi-layer routing with vias (25% of timeout)
        if time.time() - start_time < timeout * 0.95:
            via_timeout = timeout * 0.25
            
            result = self._route_with_vias_lee(
                pad_a, pad_b, net_name, net_constraints, via_timeout, start_time
            )
            
            if result:
                logger.info(f"✅ Lee's via routing succeeded")
                return result
        
        # STRATEGY 3: Fallback attempt (5% of timeout)
        if time.time() - start_time < timeout:
            logger.info(f"🔄 Lee's fallback routing for {net_name}")
            
            # Try with reduced constraints for difficult connections
            fallback_constraints = net_constraints.copy()
            fallback_constraints['clearance'] *= 0.8  # Reduce clearance slightly
            
            result = self._route_single_layer_lee(
                pad_a, pad_b, 'F.Cu', net_name, fallback_constraints,
                timeout * 0.05, start_time
            )
            
            if result:
                logger.warning(f"⚠️ Lee's fallback routing succeeded for {net_name}")
                return result
        
        logger.error(f"❌ Lee's algorithm failed to route {net_name}")
        return None
    
    def _route_single_layer_lee(self, pad_a: Dict, pad_b: Dict, layer: str, 
                               net_name: str, net_constraints: Dict, 
                               timeout: float, start_time: float) -> Optional[List[RouteSegment]]:
        """Route on a single layer using Lee's wavefront expansion"""
        
        if time.time() - start_time > timeout:
            return None
        
        # Start with base Free Routing Space (fixed obstacles only)
        working_grid = self.gpu_manager.copy_array(self.obstacle_grids[layer])
        
        # Add pre-computed pad exclusions (OPTIMIZED - no more O(N×P) bottleneck!)
        self._add_other_nets_pad_exclusions(working_grid, layer, net_name)
        
        # Clear current net's pads from obstacles (they're targets, not obstacles)
        self._exclude_net_pads_from_obstacles(working_grid, layer, net_name)
        
        # Get pad grid positions
        pad_a_gx, pad_a_gy = self.grid_config.world_to_grid(pad_a['x'], pad_a['y'])
        pad_b_gx, pad_b_gy = self.grid_config.world_to_grid(pad_b['x'], pad_b['y'])
        
        # Perform Lee's wavefront expansion
        path = self._lee_wavefront_expansion(
            working_grid, (pad_a_gx, pad_a_gy), (pad_b_gx, pad_b_gy), timeout - (time.time() - start_time)
        )
        
        if path:
            # Convert path to route segments
            segments = self._path_to_route_segments(path, layer, net_name, net_constraints)
            
            # Update obstacle grids with new routes
            self._add_route_to_obstacles(segments, working_grid, layer)
            self.obstacle_grids[layer] = working_grid
            
            return segments
        
        return None
    
    def _lee_wavefront_expansion(self, obstacle_grid, start: Tuple[int, int], 
                                end: Tuple[int, int], timeout: float) -> Optional[List[Tuple[int, int]]]:
        """Core Lee's algorithm wavefront expansion with GPU acceleration"""
        
        start_time = time.time()
        
        # Initialize distance grid
        if self.gpu_manager.is_gpu_enabled():
            distance_grid = cp.full_like(obstacle_grid, -1, dtype=cp.int32)
        else:
            distance_grid = np.full_like(obstacle_grid, -1, dtype=np.int32)
        
        # Mark start position
        distance_grid[start[1], start[0]] = 0
        
        # Current wavefront
        if self.gpu_manager.is_gpu_enabled():
            current_wave = cp.zeros_like(obstacle_grid, dtype=cp.bool_)
        else:
            current_wave = np.zeros_like(obstacle_grid, dtype=bool)
        
        current_wave[start[1], start[0]] = True
        wave_distance = 0
        
        # Wavefront expansion loop
        for iteration in range(self.max_iterations):
            if time.time() - start_time > timeout:
                logger.warning("⏰ Lee's wavefront expansion timeout")
                break
            
            # Check if we reached the target
            if distance_grid[end[1], end[0]] >= 0:
                logger.debug(f"🎯 Lee's algorithm reached target in {iteration} iterations")
                return self._backtrace_path(distance_grid, start, end)
            
            # Expand wavefront
            next_wave = self._expand_wavefront_gpu(current_wave, obstacle_grid, distance_grid, wave_distance + 1)
            
            # Check if expansion is stuck - use efficient method to avoid GPU sync
            if self.gpu_manager.is_gpu_enabled():
                # Use more efficient check - avoid cp.sum() which can cause sync
                # Check if any new cells were added by comparing wave size
                if hasattr(next_wave, 'any'):
                    if not next_wave.any():  # More efficient than sum() > 0
                        break
                else:
                    if cp.sum(next_wave) == 0:
                        break
            else:
                if np.sum(next_wave) == 0:
                    break
            
            current_wave = next_wave
            wave_distance += 1
        
        logger.warning(f"❌ Lee's algorithm failed to find path after {iteration + 1} iterations")
        return None
    
    def _expand_wavefront_gpu(self, current_wave, obstacle_grid, distance_grid, new_distance: int):
        """Expand wavefront by one step using GPU acceleration"""
        
        if self.gpu_manager.is_gpu_enabled():
            # GPU implementation using binary dilation
            # Create structuring element for 8-connected neighbors (45° routing)
            structure = cp.array([[1, 1, 1], 
                                 [1, 1, 1], 
                                 [1, 1, 1]], dtype=cp.bool_)
            
            # Dilate current wave
            from cupyx.scipy.ndimage import binary_dilation
            expanded = binary_dilation(current_wave, structure=structure, iterations=1)
            
            # Remove obstacles and already visited cells
            next_wave = expanded & ~obstacle_grid & (distance_grid == -1)
            
            # Update distance grid
            distance_grid[next_wave] = new_distance
            
            return next_wave
        else:
            # CPU implementation
            from scipy.ndimage import binary_dilation
            
            # Create structuring element for 8-connected neighbors (45° routing)
            structure = np.array([[1, 1, 1], 
                                 [1, 1, 1], 
                                 [1, 1, 1]], dtype=bool)
            
            # Dilate current wave
            expanded = binary_dilation(current_wave, structure=structure, iterations=1)
            
            # Remove obstacles and already visited cells
            next_wave = expanded & ~obstacle_grid & (distance_grid == -1)
            
            # Update distance grid
            distance_grid[next_wave] = new_distance
            
            return next_wave
    
    def _backtrace_path(self, distance_grid, start: Tuple[int, int], 
                       end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrace from end to start to find optimal path"""
        
    def _backtrace_path(self, distance_grid, start: Tuple[int, int], 
                       end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrace from end to start to find optimal path"""
        
        # Optimize GPU-CPU transfers by processing small regions at a time
        path = []
        current = end
        
        # Keep backtracing on GPU if possible to avoid expensive full grid transfer
        if self.gpu_manager.is_gpu_enabled() and hasattr(distance_grid, 'get'):
            # Process backtrace in small chunks to minimize GPU-CPU transfers
            while current != start:
                path.append(current)
                
                # Get small local region around current position (3x3 neighborhood)
                cx, cy = current
                min_x, max_x = max(0, cx-1), min(distance_grid.shape[1], cx+2)
                min_y, max_y = max(0, cy-1), min(distance_grid.shape[0], cy+2)
                
                # Transfer only this tiny region (9 cells max vs entire grid)
                local_region = distance_grid[min_y:max_y, min_x:max_x].get()
                local_cx, local_cy = cx - min_x, cy - min_y
                
                # Find best neighbor in local region
                best_distance = float('inf')
                best_neighbor = None
                
                for dx, dy in self.neighbor_offsets:
                    nx_local, ny_local = local_cx + dx, local_cy + dy
                    
                    if (0 <= nx_local < local_region.shape[1] and 
                        0 <= ny_local < local_region.shape[0] and
                        local_region[ny_local, nx_local] >= 0):
                        
                        if local_region[ny_local, nx_local] < best_distance:
                            best_distance = local_region[ny_local, nx_local]
                            best_neighbor = (cx + dx, cy + dy)
                
                if best_neighbor is None:
                    logger.error("❌ Lee's backtrace failed - no valid path")
                    return []
                
                current = best_neighbor
        else:
            # CPU fallback - convert once if needed
            if hasattr(distance_grid, 'get'):
                distance_grid_cpu = distance_grid.get()
            else:
                distance_grid_cpu = distance_grid
                
            while current != start:
                path.append(current)
                
                # Find neighbor with lowest distance
                cx, cy = current
                best_distance = float('inf')
                best_neighbor = None
                
                for dx, dy in self.neighbor_offsets:
                    nx, ny = cx + dx, cy + dy
                    
                    if (0 <= nx < distance_grid_cpu.shape[1] and 
                        0 <= ny < distance_grid_cpu.shape[0] and
                        distance_grid_cpu[ny, nx] >= 0):
                        
                        if distance_grid_cpu[ny, nx] < best_distance:
                            best_distance = distance_grid_cpu[ny, nx]
                            best_neighbor = (nx, ny)
                
                if best_neighbor is None:
                    logger.error("❌ Lee's backtrace failed - no valid path")
                    return []
                
                current = best_neighbor
        
        path.append(start)
        path.reverse()
        
        logger.debug(f"🛤️ Lee's algorithm found path with {len(path)} points")
        return path
    
    def _route_with_vias_lee(self, pad_a: Dict, pad_b: Dict, net_name: str, 
                            net_constraints: Dict, timeout: float, start_time: float) -> Optional[List[RouteSegment]]:
        """Route using vias to connect between layers"""
        
        # Try strategic via positions
        via_positions = self._get_strategic_via_positions(pad_a, pad_b)
        
        for i, (via_x, via_y) in enumerate(via_positions):
            if time.time() - start_time > timeout * (i + 1) / len(via_positions):
                break
            
            logger.debug(f"🔗 Trying via at ({via_x:.1f}, {via_y:.1f})")
            
            # Try routing: pad_a -> via -> pad_b
            result = self._route_with_via_at(
                pad_a, pad_b, via_x, via_y, net_name, net_constraints, 
                timeout / len(via_positions), start_time
            )
            
            if result:
                logger.info(f"✅ Lee's via routing succeeded with via at position {i+1}")
                return result
        
        return None
    
    def _route_with_via_at(self, pad_a: Dict, pad_b: Dict, via_x: float, via_y: float,
                          net_name: str, net_constraints: Dict, timeout: float, start_time: float) -> Optional[List[RouteSegment]]:
        """Route through a specific via position"""
        
        remaining_timeout = timeout - (time.time() - start_time)
        if remaining_timeout <= 0:
            return None
        
        # Create virtual via pad
        via_pad = {
            'x': via_x, 
            'y': via_y, 
            'size_x': net_constraints['via_size'], 
            'size_y': net_constraints['via_size'],
            'net': net_name
        }
        
        # Route first segment (pad_a to via)
        segment1 = self._route_single_layer_lee(
            pad_a, via_pad, 'F.Cu', net_name, net_constraints, 
            remaining_timeout * 0.4, start_time
        )
        
        if not segment1:
            return None
        
        # Route second segment (via to pad_b)
        segment2 = self._route_single_layer_lee(
            via_pad, pad_b, 'B.Cu', net_name, net_constraints,
            remaining_timeout * 0.4, start_time
        )
        
        if not segment2:
            return None
        
        # Create via segment
        via_segment = RouteSegment(
            type='via',
            start_x=via_x,
            start_y=via_y,
            width=net_constraints['via_size'],
            net_name=net_name
        )
        
        # Combine all segments
        all_segments = segment1 + [via_segment] + segment2
        
        return all_segments
    
    def _route_multi_pad_net_lee(self, pads: List[Dict], net_name: str, 
                                net_constraints: Dict, timeout: float, start_time: float) -> Optional[List[RouteSegment]]:
        """Route multi-pad net using minimum spanning tree approach"""
        
        if len(pads) < 2:
            return None
        
        logger.info(f"🌳 Lee's MST routing for {net_name} with {len(pads)} pads")
        
        # Build minimum spanning tree of connections
        connections = self._build_minimum_spanning_tree(pads)
        
        all_segments = []
        per_connection_timeout = timeout / len(connections)
        
        # Route each connection in the MST
        for i, (pad_a_idx, pad_b_idx) in enumerate(connections):
            remaining_timeout = timeout - (time.time() - start_time)
            if remaining_timeout <= 0:
                logger.warning("⏰ Multi-pad routing timeout")
                break
            
            pad_a = pads[pad_a_idx]
            pad_b = pads[pad_b_idx]
            
            connection_segments = self._route_two_pads_lee(
                pad_a, pad_b, net_name, net_constraints, 
                min(per_connection_timeout, remaining_timeout), start_time
            )
            
            if connection_segments:
                all_segments.extend(connection_segments)
                logger.debug(f"✅ MST connection {i+1}/{len(connections)} routed")
            else:
                logger.error(f"❌ Failed MST connection {i+1}/{len(connections)}")
                return None
        
        logger.info(f"✅ Lee's MST routing completed: {len(all_segments)} segments")
        return all_segments
    
    def _build_minimum_spanning_tree(self, pads: List[Dict]) -> List[Tuple[int, int]]:
        """Build minimum spanning tree for multi-pad net"""
        if len(pads) < 2:
            return []
        
        # Calculate all pairwise distances
        distances = {}
        for i in range(len(pads)):
            for j in range(i + 1, len(pads)):
                dist = self.board_interface.calculate_connection_distance(pads[i], pads[j])
                distances[(i, j)] = dist
        
        # Kruskal's algorithm for MST
        edges = sorted(distances.items(), key=lambda x: x[1])  # Sort by distance
        parent = list(range(len(pads)))
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
                return True
            return False
        
        mst_edges = []
        for (i, j), dist in edges:
            if union(i, j):
                mst_edges.append((i, j))
                if len(mst_edges) == len(pads) - 1:
                    break
        
        return mst_edges
    
    def _select_best_layer_for_connection(self, pad_a: Dict, pad_b: Dict) -> str:
        """Select the best layer for routing based on obstacle density"""
        
        # Simple heuristic: choose layer with lower obstacle density in connection area
        pad_a_gx, pad_a_gy = self.grid_config.world_to_grid(pad_a['x'], pad_a['y'])
        pad_b_gx, pad_b_gy = self.grid_config.world_to_grid(pad_b['x'], pad_b['y'])
        
        # Define region of interest
        min_gx = min(pad_a_gx, pad_b_gx) - 10
        max_gx = max(pad_a_gx, pad_b_gx) + 10
        min_gy = min(pad_a_gy, pad_b_gy) - 10
        max_gy = max(pad_a_gy, pad_b_gy) + 10
        
        # Clip to grid bounds
        min_gx = max(0, min_gx)
        max_gx = min(self.grid_config.width - 1, max_gx)
        min_gy = max(0, min_gy)
        max_gy = min(self.grid_config.height - 1, max_gy)
        
        best_layer = 'F.Cu'  # Default
        min_density = float('inf')
        
        for layer in self.layers:
            grid = self.obstacle_grids[layer]
            
            # Optimize GPU usage - avoid full grid transfer for density calculation
            if self.gpu_manager.is_gpu_enabled() and hasattr(grid, 'get'):
                # Only transfer the small region of interest instead of entire grid
                region_gpu = grid[min_gy:max_gy+1, min_gx:max_gx+1]
                region_cpu = region_gpu.get()  # Much smaller transfer
                density = np.sum(region_cpu) / region_cpu.size if region_cpu.size > 0 else 0
            else:
                # CPU case - direct region access
                grid_cpu = self.gpu_manager.to_cpu(grid) if hasattr(grid, 'get') else grid
                region = grid_cpu[min_gy:max_gy+1, min_gx:max_gx+1]
                density = np.sum(region) / region.size if region.size > 0 else 0
            
            if density < min_density:
                min_density = density
                best_layer = layer
        
        logger.debug(f"🎯 Selected {best_layer} for routing (density: {min_density:.3f})")
        return best_layer
    
    def _get_strategic_via_positions(self, pad_a: Dict, pad_b: Dict) -> List[Tuple[float, float]]:
        """Get strategic positions for via placement"""
        
        src_x, src_y = pad_a['x'], pad_a['y']
        tgt_x, tgt_y = pad_b['x'], pad_b['y']
        
        # Calculate connection vector
        dx = tgt_x - src_x
        dy = tgt_y - src_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance < 0.1:  # Very close pads
            return [(src_x, src_y)]
        
        positions = [
            (src_x + dx * 0.5, src_y + dy * 0.5),    # Midpoint
            (src_x + dx * 0.3, src_y + dy * 0.3),    # 30% from source
            (src_x + dx * 0.7, src_y + dy * 0.7),    # 70% from source
        ]
        
        # Add perpendicular offsets for obstacle avoidance
        if distance > 1.0:  # Only for longer connections
            perp_dx, perp_dy = -dy, dx  # Perpendicular vector
            if distance > 0:
                perp_dx, perp_dy = perp_dx / distance, perp_dy / distance
            
            offset_distance = min(1.0, distance * 0.3)  # 30% of connection distance or 1mm max
            positions.extend([
                (src_x + dx * 0.5 + perp_dx * offset_distance, src_y + dy * 0.5 + perp_dy * offset_distance),
                (src_x + dx * 0.5 - perp_dx * offset_distance, src_y + dy * 0.5 - perp_dy * offset_distance),
            ])
        
        return positions
    
    def _path_to_route_segments(self, path: List[Tuple[int, int]], layer: str, 
                               net_name: str, net_constraints: Dict) -> List[RouteSegment]:
        """Convert grid path to optimized route segments"""
        
        if len(path) < 2:
            return []
        
        # First, optimize the path by removing unnecessary intermediate points
        optimized_path = self._optimize_path_for_routing(path)
        
        segments = []
        trace_width = net_constraints.get('trace_width', self.drc_rules.default_trace_width)
        
        # Create segments between key waypoints only (not every grid cell)
        for i in range(len(optimized_path) - 1):
            gx1, gy1 = optimized_path[i]
            gx2, gy2 = optimized_path[i + 1]
            
            # Convert to world coordinates
            x1, y1 = self.grid_config.grid_to_world(gx1, gy1)
            x2, y2 = self.grid_config.grid_to_world(gx2, gy2)
            
            segment = RouteSegment(
                type='track',
                start_x=x1,
                start_y=y1,
                end_x=x2,
                end_y=y2,
                width=trace_width,
                layer=layer,
                net_name=net_name
            )
            
            segments.append(segment)
        
        logger.debug(f"🎯 Path optimization: {len(path)} grid points → {len(optimized_path)} waypoints → {len(segments)} segments")
        return segments
    
    def _optimize_path_for_routing(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Optimize path by removing unnecessary waypoints and creating efficient segments"""
        
        if len(path) < 3:
            return path
        
        optimized = [path[0]]  # Always keep start point
        
        for i in range(1, len(path) - 1):
            prev_point = optimized[-1]
            current_point = path[i]
            next_point = path[i + 1]
            
            # Check if we can skip this point (if it's on a straight line)
            if not self._is_path_change_significant(prev_point, current_point, next_point):
                continue  # Skip this intermediate point
            
            # This is a significant direction change - keep it
            optimized.append(current_point)
        
        optimized.append(path[-1])  # Always keep end point
        
        return optimized
    
    def _is_path_change_significant(self, prev: Tuple[int, int], current: Tuple[int, int], 
                                   next_point: Tuple[int, int]) -> bool:
        """Determine if a path point represents a significant direction change"""
        
        # Calculate direction vectors
        dx1, dy1 = current[0] - prev[0], current[1] - prev[1]
        dx2, dy2 = next_point[0] - current[0], next_point[1] - current[1]
        
        # If directions are the same (collinear), this point can be skipped
        # For 45-degree routing, we want to preserve diagonal vs orthogonal changes
        if (dx1, dy1) == (dx2, dy2):
            return False  # Same direction - can skip
        
        # Different directions - this is a significant waypoint
        return True
    
    def _precompute_pad_exclusion_grids(self):
        """Pre-compute pad exclusion grids for all layers to eliminate O(N×P) bottleneck
        
        This method runs once at initialization and creates cached grids with ALL pad 
        exclusions. During routing, we copy the appropriate grid and clear only the 
        current net's pads, changing complexity from O(N×P) to O(P) + O(current_net_pads).
        """
        logger.info("⚡ Pre-computing pad exclusion grids to eliminate performance bottleneck...")
        
        all_pads = self.board_interface.get_all_pads()
        
        for layer in self.layers:
            logger.debug(f"⚡ Pre-computing pad exclusions for {layer}...")
            
            # Start with copy of base Free Routing Space (obstacles only, no pads)
            pad_exclusion_grid = self.gpu_manager.copy_array(self.obstacle_grids[layer])
            
            excluded_count = 0
            
            # Add ALL pad exclusions for this layer (we'll clear current net's pads later)
            for pad in all_pads:
                if not self.board_interface.is_pad_on_layer(pad, layer):
                    continue
                
                # Add DRC-compliant exclusion for this pad
                geometry = self.board_interface.get_pad_geometry(pad)
                
                # Use DRC clearance to ensure proper spacing
                drc_clearance = self.drc_rules.min_trace_spacing  # 0.508mm
                exclusion_x = geometry['size_x'] + 2 * drc_clearance
                exclusion_y = geometry['size_y'] + 2 * drc_clearance
                
                self._mark_rectangular_obstacle(
                    pad_exclusion_grid,
                    geometry['x'], geometry['y'],
                    exclusion_x, exclusion_y
                )
                excluded_count += 1
            
            # Cache the grid for this layer
            self.pad_exclusion_grids[layer] = pad_exclusion_grid
            
            logger.debug(f"⚡ {layer}: Pre-computed {excluded_count} pad exclusions")
        
        logger.info(f"⚡ Pre-computed pad exclusions for {len(self.layers)} layers - O(N×P) bottleneck eliminated!")

    def _add_other_nets_pad_exclusions(self, obstacle_grid, layer: str, current_net: str):
        """Add pad exclusions for all nets OTHER than the current net - OPTIMIZED VERSION
        
        Uses pre-computed pad exclusion grids to eliminate O(N×P) performance bottleneck.
        Instead of processing ALL pads for every net, we copy the cached grid and clear
        only the current net's pads.
        """
        # Copy pre-computed pad exclusions for this layer (includes ALL pads)
        cached_exclusions = self.pad_exclusion_grids[layer]
        
        if self.gpu_manager.is_gpu_enabled():
            # GPU: Bitwise OR to combine base obstacles with cached pad exclusions
            if hasattr(cached_exclusions, 'copy'):
                obstacle_grid |= cached_exclusions
            else:
                # Fallback for different GPU array types
                temp = self.gpu_manager.copy_array(cached_exclusions)
                obstacle_grid |= temp
        else:
            # CPU: Simple bitwise OR
            obstacle_grid |= cached_exclusions
        
        # Now clear the current net's pads (they should be accessible as targets)
        net_pads = self.board_interface.get_pads_for_net(current_net)
        cleared_count = 0
        
        for pad in net_pads:
            if self.board_interface.is_pad_on_layer(pad, layer):
                # Clear this net's pad from exclusions (they're targets, not obstacles)
                geometry = self.board_interface.get_pad_geometry(pad)
                
                # Clear with DRC clearance area (same size as exclusion)
                drc_clearance = self.drc_rules.min_trace_spacing
                clear_x = geometry['size_x'] + 2 * drc_clearance
                clear_y = geometry['size_y'] + 2 * drc_clearance
                
                self._clear_rectangular_area(
                    obstacle_grid,
                    geometry['x'], geometry['y'],
                    clear_x, clear_y
                )
                cleared_count += 1
        
        logger.debug(f"⚡ Used cached exclusions, cleared {cleared_count} pads for net {current_net} on {layer}")
    
    def _exclude_net_pads_from_obstacles(self, obstacle_grid, layer: str, net_name: str):
        """Clear connection points for current net from obstacle grid
        
        In Free Routing Space architecture, the obstacle grid already has proper
        DRC clearances built in. We only need to clear small connection areas
        around the current net's pads to allow the router to connect to them.
        """
        print(f"🧹 DEBUG: Clearing connection points for net '{net_name}' on {layer}")
        
        # Find pads for this net
        net_pads = self.board_interface.get_pads_for_net(net_name)
        print(f"🧹 DEBUG: Found {len(net_pads)} pads for net {net_name}")
        
        cleared_count = 0
        for pad in net_pads:
            if self.board_interface.is_pad_on_layer(pad, layer):
                print(f"🧹 DEBUG: Clearing connection point for pad on {layer}")
                
                # Get pad geometry
                geometry = self.board_interface.get_pad_geometry(pad)
                
                # Clear MINIMAL connection area - just enough to connect
                # The Free Routing Space already handles DRC compliance
                connection_clearance = 0.2  # Minimal 0.2mm clearance for connection
                clear_size_x = geometry['size_x'] + 2 * connection_clearance
                clear_size_y = geometry['size_y'] + 2 * connection_clearance
                
                print(f"🧹 DEBUG: Clearing {clear_size_x:.2f}x{clear_size_y:.2f} connection area")
                
                self._clear_rectangular_area(
                    obstacle_grid, 
                    geometry['x'], geometry['y'],
                    clear_size_x, clear_size_y
                )
                cleared_count += 1
        
        print(f"🧹 DEBUG: Cleared {cleared_count} connection points for net {net_name} on {layer}")
        
        # NOTE: No need for _protect_other_nets_pads() in Free Routing Space architecture!
        # The Free Routing Space already has proper DRC clearances built in.
    
    def _mark_rectangular_area_as_obstacle(self, obstacle_grid, center_x: float, center_y: float, 
                                         size_x: float, size_y: float):
        """Mark a rectangular area as obstacle in the grid - DEPRECATED
        
        This method is kept for compatibility but should not be needed
        in the Free Routing Space architecture.
        """
        logger.warning("⚠️ _mark_rectangular_area_as_obstacle called - this should not be needed in Free Routing Space architecture")
        
        # Keep implementation for compatibility
        # Convert to grid coordinates
        grid_x, grid_y = self.grid_config.world_to_grid(center_x, center_y)
        
        # Calculate grid extents
        half_size_x_cells = max(1, int(math.ceil((size_x / 2) / self.grid_config.resolution)))
        half_size_y_cells = max(1, int(math.ceil((size_y / 2) / self.grid_config.resolution)))
        
        # Mark the rectangular area as obstacle
        for dx in range(-half_size_x_cells, half_size_x_cells + 1):
            for dy in range(-half_size_y_cells, half_size_y_cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                
                if self.grid_config.is_valid_grid_position(gx, gy):
                    obstacle_grid[gy, gx] = True
    
    def _clear_rectangular_area(self, obstacle_grid, center_x: float, center_y: float, 
                               size_x: float, size_y: float):
        """Clear a rectangular area from the obstacle grid"""
        
        # Convert to grid coordinates
        grid_x, grid_y = self.grid_config.world_to_grid(center_x, center_y)
        
        # Calculate grid extents
        half_size_x_cells = max(1, int(math.ceil((size_x / 2) / self.grid_config.resolution)))
        half_size_y_cells = max(1, int(math.ceil((size_y / 2) / self.grid_config.resolution)))
        
        # Clear the rectangular area
        for dx in range(-half_size_x_cells, half_size_x_cells + 1):
            for dy in range(-half_size_y_cells, half_size_y_cells + 1):
                gx, gy = grid_x + dx, grid_y + dy
                
                if self.grid_config.is_valid_grid_position(gx, gy):
                    obstacle_grid[gy, gx] = False
    
    def _add_route_to_obstacles(self, segments: List[RouteSegment], obstacle_grid, layer: str):
        """Add routed segments to obstacle grid for future routing attempts"""
        
        for segment in segments:
            if segment.type == 'track' and segment.layer == layer:
                # Mark track area as obstacle with proper clearance
                self._mark_line_obstacle(
                    obstacle_grid,
                    segment.start_x, segment.start_y,
                    segment.end_x, segment.end_y,
                    segment.width
                )
            elif segment.type == 'via':
                # Mark via area as obstacle
                self._mark_circular_obstacle(
                    obstacle_grid, segment.start_x, segment.start_y, segment.width / 2
                )
