"""
True Parallel PathFinder Implementation
Routes all nets simultaneously, then negotiates congestion iteratively
"""

import cupy as cp
import logging
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParallelRouteRequest:
    """Request for parallel routing"""
    net_id: str
    source_idx: int
    sink_idx: int
    priority: float = 1.0

class TrueParallelPathFinder:
    """True PathFinder - routes all nets simultaneously with negotiated congestion"""
    
    def __init__(self, gpu_rrg):
        self.gpu_rrg = gpu_rrg
        self.max_iterations = 50
        self.pres_fac_base = 1.0
        self.pres_fac_mult = 1.5
        self.acc_fac = 1.0
        self.progress_callback = None  # Callback for GUI updates
        
    def route_all_nets_parallel(self, requests: List[ParallelRouteRequest]) -> Dict[str, List[int]]:
        """Route all nets simultaneously using true PathFinder algorithm with timeout detection"""
        
        logger.info(f"Starting TRUE PARALLEL PathFinder for {len(requests)} nets")
        
        # Handle empty requests
        if not requests:
            logger.warning("No route requests provided to parallel PathFinder")
            return {}
        
        # Initialize congestion tracking
        self._initialize_congestion_state(len(requests))
        
        iteration = 0
        routes = {}
        start_time = time.time()
        last_progress_time = start_time
        max_routing_time = 300  # 5 minutes timeout
        iteration_timeout = 30  # 30 seconds per iteration
        
        while iteration < self.max_iterations:
            iteration_start = time.time()
            
            # Check global timeout
            if time.time() - start_time > max_routing_time:
                logger.warning(f"PathFinder global timeout after {max_routing_time}s - stopping at iteration {iteration + 1}")
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'timeout',
                        'message': f"Routing timeout after {max_routing_time}s"
                    })
                break
            logger.info(f"PathFinder iteration {iteration + 1}/{self.max_iterations}")
            
            # Update GUI progress if callback available
            if self.progress_callback:
                self.progress_callback({
                    'type': 'iteration_start',
                    'iteration': iteration + 1,
                    'max_iterations': self.max_iterations,
                    'status': f"Routing iteration {iteration + 1}..."
                })
            
            # PHASE 1: Route ALL nets simultaneously (ignoring conflicts)
            iteration_routes = self._route_all_nets_simultaneously(requests)
            
            # Check iteration timeout
            iteration_time = time.time() - iteration_start
            if iteration_time > iteration_timeout:
                logger.warning(f"Iteration {iteration + 1} timeout after {iteration_time:.1f}s (max {iteration_timeout}s)")
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'iteration_timeout',
                        'iteration': iteration + 1,
                        'time': iteration_time
                    })
                # Continue with current routes but increase timeout for next iteration
                iteration_timeout = min(60, iteration_timeout * 1.5)
            
            # PHASE 2: Calculate congestion from all routes
            congestion_map = self._calculate_congestion(iteration_routes)
            
            # PHASE 3: Update costs based on congestion
            self._update_congestion_costs(congestion_map, iteration)
            
            # Update GUI with routing results and timing
            if self.progress_callback:
                successful_routes = len([r for r in iteration_routes.values() if r])
                failed_routes = len(iteration_routes) - successful_routes
                congested_edges = int(cp.sum(congestion_map > 1)) if hasattr(congestion_map, 'sum') else 0
                total_edges = len(congestion_map) if hasattr(congestion_map, '__len__') else 1
                
                self.progress_callback({
                    'type': 'routing_update',
                    'successful_routes': successful_routes,
                    'failed_routes': failed_routes,
                    'congested_edges': congested_edges,
                    'total_edges': total_edges,
                    'iteration_time': iteration_time,
                    'total_time': time.time() - start_time
                })
            
            # PHASE 4: Check convergence
            if self._check_convergence(congestion_map):
                logger.info(f"PathFinder converged after {iteration + 1} iterations")
                routes = iteration_routes
                
                if self.progress_callback:
                    self.progress_callback({
                        'type': 'convergence',
                        'iteration': iteration + 1,
                        'status': f"Converged after {iteration + 1} iterations!"
                    })
                break
                
            routes = iteration_routes
            iteration += 1
        
        successful_count = len([r for r in routes.values() if r])
        logger.info(f"PathFinder completed: {successful_count}/{len(requests)} successful routes")
        
        # Final update
        if self.progress_callback:
            self.progress_callback({
                'type': 'completion',
                'successful_routes': successful_count,
                'total_routes': len(requests),
                'converged': iteration < self.max_iterations
            })
        
        return routes
    
    def _route_all_nets_simultaneously(self, requests: List[ParallelRouteRequest]) -> Dict[str, List[int]]:
        """Route all nets in parallel - the core of PathFinder"""
        
        # Handle empty requests
        if not requests:
            logger.warning("No route requests provided to parallel PathFinder")
            return {}
        
        # Create GPU arrays for batch processing
        num_nets = len(requests)
        source_indices = cp.array([req.source_idx for req in requests], dtype=cp.int32)
        sink_indices = cp.array([req.sink_idx for req in requests], dtype=cp.int32)
        
        # Debug source/sink indices
        logger.info(f"PathFinder source indices range: {int(cp.min(source_indices))} to {int(cp.max(source_indices))}")
        logger.info(f"PathFinder sink indices range: {int(cp.min(sink_indices))} to {int(cp.max(sink_indices))}")
        logger.info(f"Total routing nodes available: {self.gpu_rrg.num_nodes}")
        
        # Debug tap node usage
        if hasattr(self.gpu_rrg, 'tap_nodes'):
            tap_node_indices = set(self.gpu_rrg.tap_nodes.values())
            source_tap_count = sum(1 for idx in source_indices if int(idx) in tap_node_indices)
            sink_tap_count = sum(1 for idx in sink_indices if int(idx) in tap_node_indices)
            logger.info(f"TAP NODE USAGE: {source_tap_count}/{len(source_indices)} sources are tap nodes")
            logger.info(f"TAP NODE USAGE: {sink_tap_count}/{len(sink_indices)} sinks are tap nodes")
            if source_tap_count == 0 and sink_tap_count == 0:
                logger.error("NO TAP NODES IN ROUTING - this explains why paths aren't found!")
        
        # Validate indices are within bounds
        invalid_sources = cp.sum((source_indices < 0) | (source_indices >= self.gpu_rrg.num_nodes))
        invalid_sinks = cp.sum((sink_indices < 0) | (sink_indices >= self.gpu_rrg.num_nodes))
        if invalid_sources > 0 or invalid_sinks > 0:
            logger.error(f"Invalid indices: {invalid_sources} invalid sources, {invalid_sinks} invalid sinks")
            return {}
        
        # Initialize distance arrays for all nets
        num_nodes = self.gpu_rrg.num_nodes
        distances = cp.full((num_nets, num_nodes), cp.inf, dtype=cp.float32)
        parents = cp.full((num_nets, num_nodes), -1, dtype=cp.int32)
        visited = cp.zeros((num_nets, num_nodes), dtype=cp.bool_)
        
        # Set source distances to 0
        net_indices = cp.arange(num_nets)
        distances[net_indices, source_indices] = 0.0
        
        # Parallel wavefront expansion for all nets
        active_frontiers = cp.zeros((num_nets, num_nodes), dtype=cp.bool_)
        active_frontiers[net_indices, source_indices] = True
        
        # Debug initial frontier setup
        initial_active_count = int(cp.sum(active_frontiers))
        logger.info(f"Initial active frontier nodes: {initial_active_count} (should be {num_nets})")
        if initial_active_count != num_nets:
            logger.error("Frontier initialization failed - some sources not activated")
        
        max_steps = 5000  # Increased for large boards (200mm+ routing)
        step = 0
        last_progress_step = 0
        step_start_time = time.time()
        
        while cp.any(active_frontiers) and step < max_steps:
            # Debug main loop conditions for first few steps
            if step <= 5:
                active_count = int(cp.sum(active_frontiers))
                logger.error(f"MAIN LOOP Step {step}: {active_count} active frontiers, max_steps={max_steps}")
            
            # Expand all active frontiers simultaneously
            self._expand_all_frontiers_parallel(active_frontiers, distances, parents, visited, step)
            
            # Check if any sinks reached
            sinks_reached = visited[net_indices, sink_indices]
            routed_now = int(cp.sum(sinks_reached))
            
            if step <= 5:
                logger.error(f"  After expansion: {routed_now}/{num_nets} sinks reached")
                
            if cp.all(sinks_reached):
                logger.info(f"All nets routed in {step} wavefront steps")
                break
                
            # Check if we have active frontiers for next iteration
            active_count_after = int(cp.sum(active_frontiers))
            if step <= 5:
                logger.error(f"  Active frontiers after expansion: {active_count_after}")
                
            if active_count_after == 0:
                logger.error(f"TERMINATION: No active frontiers remaining at step {step} (routed {routed_now}/{num_nets})")
                break
            
            # Progress monitoring every 100 steps
            if step > 0 and step % 100 == 0:
                routed_count = int(cp.sum(sinks_reached))
                step_time = time.time() - step_start_time
                logger.info(f"Wavefront step {step}: {routed_count}/{num_nets} nets routed in {step_time:.2f}s")
                
                # Check for stuck progress
                if routed_count == last_progress_step:
                    logger.warning(f"No routing progress in last 100 steps (step {step})")
                    # Don't break - some nets might still be expanding
                
                last_progress_step = routed_count
                step_start_time = time.time()
                
            step += 1
            
        if step >= max_steps:
            logger.warning(f"Wavefront expansion reached maximum steps ({max_steps}) - some nets may be unrouted")
        
        # Reconstruct paths for all nets
        routes = {}
        for i, req in enumerate(requests):
            if visited[i, req.sink_idx]:
                path = self._reconstruct_path_parallel(i, req.source_idx, req.sink_idx, parents)
                routes[req.net_id] = path
            else:
                routes[req.net_id] = []  # Failed to route
                
        return routes
    
    def _expand_all_frontiers_parallel(self, frontiers, distances, parents, visited, step):
        """Expand wavefronts for all nets simultaneously using GPU CSR operations"""
        
        # Check for empty frontiers first
        if not cp.any(frontiers):
            logger.warning("No active frontiers - all nets may be routed or stuck")
            return
            
        # Find minimum distance nodes in each frontier
        frontier_distances = cp.where(frontiers, distances, cp.inf)
        
        # Handle case where some nets have no active frontier
        try:
            min_distances = cp.min(frontier_distances, axis=1, keepdims=True)
            
            # DEBUG: Log shapes and values for first few steps
            if step <= 2:
                logger.error(f"SHAPE DEBUG Step {step}: frontier_distances {frontier_distances.shape}")
                logger.error(f"SHAPE DEBUG Step {step}: min_distances {min_distances.shape}")
                logger.error(f"SHAPE DEBUG Step {step}: distances {distances.shape}")
                # Sample values from first net
                if frontiers.shape[0] > 0:
                    sample_frontier = frontiers[0]
                    sample_distances = distances[0]
                    frontier_indices = cp.where(sample_frontier)[0][:3]  # First 3 frontier nodes
                    if len(frontier_indices) > 0:
                        logger.error(f"SHAPE DEBUG: Net 0 has {int(cp.sum(sample_frontier))} frontier nodes")
                        logger.error(f"SHAPE DEBUG: Net 0 frontier sample distances: {sample_distances[frontier_indices].tolist()}")
                        logger.error(f"SHAPE DEBUG: Net 0 min_distance: {float(min_distances[0, 0])}")
        except ValueError as e:
            logger.warning(f"Error computing min distances: {e} - skipping expansion")
            return
        
        # Debug: Check shapes to prevent broadcasting errors
        if min_distances.shape[0] != distances.shape[0]:
            logger.warning(f"Shape mismatch: min_distances {min_distances.shape} vs distances {distances.shape}")
            # If we have fewer minimum distances than nets, pad with inf
            if min_distances.shape[0] < distances.shape[0]:
                padding = cp.full((distances.shape[0] - min_distances.shape[0], 1), cp.inf, dtype=cp.float32)
                min_distances = cp.vstack([min_distances, padding])
            else:
                # Reshape min_distances to match number of nets
                min_distances = min_distances[:distances.shape[0]].reshape(distances.shape[0], -1)
        
        # Select nodes at minimum distance for each net (with tolerance for floating point precision)
        tolerance = 1e-6
        
        # Ensure min_distances broadcasts correctly with distances [num_nets, num_nodes]
        if min_distances.shape[1] == 1:
            # min_distances is [num_nets, 1], broadcast to [num_nets, num_nodes]
            min_distances_broadcasted = cp.broadcast_to(min_distances, distances.shape)
        else:
            min_distances_broadcasted = min_distances
        
        current_nodes = frontiers & (cp.abs(distances - min_distances_broadcasted) < tolerance)
        
        # DEBUG: Check if node selection is failing
        total_frontier = int(cp.sum(frontiers))
        total_selected = int(cp.sum(current_nodes))
        
        # Always log for first 10 steps to debug issues
        if step <= 10:
            logger.error(f"STEP {step}: {total_frontier} frontier -> {total_selected} selected nodes")
            
            if total_selected == 0:
                logger.error(f"CRITICAL: ZERO NODES SELECTED FOR EXPANSION at step {step}!")
                
                # Sample detailed debugging for first net
                if total_frontier > 0:
                    sample_net = 0
                    frontier_mask = frontiers[sample_net]
                    if cp.any(frontier_mask):
                        frontier_indices = cp.where(frontier_mask)[0][:5]  # First 5 frontier nodes
                        sample_distances = distances[sample_net, frontier_indices]
                        sample_min = min_distances[sample_net, 0] if min_distances.shape[1] > 0 else "INVALID"
                        sample_min_broadcast = min_distances_broadcasted[sample_net, frontier_indices]
                        
                        logger.error(f"  Net 0 frontier distances: {sample_distances.tolist()}")
                        logger.error(f"  Net 0 min_distance: {sample_min}")  
                        logger.error(f"  Broadcasted mins: {sample_min_broadcast.tolist()}")
                        logger.error(f"  Differences: {cp.abs(sample_distances - sample_min_broadcast).tolist()}")
                        logger.error(f"  Tolerance check: {(cp.abs(sample_distances - sample_min_broadcast) < tolerance).tolist()}")
                        logger.error(f"  Tolerance value: {tolerance}")
                        
                        # Check if distances are all inf
                        all_distances = distances[sample_net]
                        finite_distances = cp.isfinite(all_distances)
                        logger.error(f"  {int(cp.sum(finite_distances))}/{all_distances.shape[0]} distances are finite")
                        
                        if cp.any(finite_distances):
                            finite_dist_vals = all_distances[finite_distances][:10]
                            logger.error(f"  Sample finite distances: {finite_dist_vals.tolist()}")
                
                # Check if this is termination condition
                if total_frontier == 0:
                    logger.error(f"  TERMINATION: No frontier nodes remaining at step {step}")
                    # Will be handled by the main while loop condition
            else:
                logger.info(f"  Step {step} OK: Selected {total_selected} nodes from {total_frontier} frontier nodes")
        
        # Mark selected nodes as visited
        visited |= current_nodes
        frontiers &= ~current_nodes  # Remove from frontier
        
        # For each net and each current node, expand neighbors
        # Use the appropriate adjacency matrix (GPU uses adjacency_csr, CPU uses adjacency_matrix)
        if hasattr(self.gpu_rrg, 'adjacency_csr') and self.gpu_rrg.adjacency_csr is not None:
            adj_csr = self.gpu_rrg.adjacency_csr
        else:
            adj_csr = self.gpu_rrg.adjacency_matrix
        
        # Debug: Check if we have current nodes to expand
        total_current = int(cp.sum(current_nodes))
        if total_current == 0 and step < 10:  # Only log for first few steps
            logger.warning(f"No current nodes to expand in step {step}")
        
        # This is the tricky part - parallel neighbor expansion for multiple nets
        nodes_expanded = 0
        neighbors_found = 0
        tap_nodes_expanded = 0
        tap_neighbors_found = 0
        
        for net_idx in range(current_nodes.shape[0]):
            current_net_nodes = cp.where(current_nodes[net_idx])[0]
            
            for node_idx in current_net_nodes:
                node_idx = int(node_idx)
                
                nodes_expanded += 1
                    
                # Get neighbors - check both CSR matrix and tap connections
                csr_neighbors = []
                tap_neighbors = []
                
                # Get CSR neighbors (if node is within CSR bounds)
                if node_idx < len(adj_csr.indptr) - 1:
                    start_idx = int(adj_csr.indptr[node_idx])
                    end_idx = int(adj_csr.indptr[node_idx + 1])
                    if start_idx < end_idx:
                        csr_neighbors = list(adj_csr.indices[start_idx:end_idx])
                
                # Get tap neighbors (separate storage for tap connections)
                if hasattr(self.gpu_rrg, 'tap_edge_connections') and node_idx in self.gpu_rrg.tap_edge_connections:
                    tap_connections = self.gpu_rrg.tap_edge_connections[node_idx]
                    tap_neighbors = [conn['to_node'] for conn in tap_connections]
                
                # Combine all neighbors
                all_neighbors = csr_neighbors + tap_neighbors
                neighbor_count = len(all_neighbors)
                
                # DEBUG: Log neighbor lookup for first few nodes in early steps
                if step <= 1 and nodes_expanded <= 5:
                    csr_count = len(csr_neighbors)
                    tap_count = len(tap_neighbors)
                    has_tap_connections = hasattr(self.gpu_rrg, 'tap_edge_connections')
                    is_in_tap_connections = node_idx in self.gpu_rrg.tap_edge_connections if has_tap_connections else False
                    logger.error(f"  NEIGHBOR DEBUG node {node_idx}: CSR={csr_count}, tap={tap_count}, has_attr={has_tap_connections}, in_tap={is_in_tap_connections}")
                    if csr_count == 0 and tap_count == 0:
                        logger.error(f"    ISOLATED: Node {node_idx} has no neighbors in either CSR or tap connections")
                
                # Check if this is a tap node and debug its connectivity
                is_tap_node = False
                if hasattr(self.gpu_rrg, 'tap_nodes'):
                    for tap_id, tap_node_idx in self.gpu_rrg.tap_nodes.items():
                        if tap_node_idx == node_idx:
                            is_tap_node = True
                            tap_nodes_expanded += 1
                            tap_neighbors_found += neighbor_count
                            if step <= 5:  # Only log first few steps to avoid spam
                                logger.info(f"GRAPH CONNECTIVITY: Expanding tap node {tap_id} (idx {node_idx}) with {neighbor_count} neighbors")
                            break
                
                if neighbor_count > 0:
                    neighbors_found += neighbor_count
                
                if neighbor_count > 0:
                    # Create combined neighbor arrays from both CSR and tap sources
                    if csr_neighbors and tap_neighbors:
                        # Both CSR and tap neighbors exist
                        neighbor_indices = cp.array(csr_neighbors + tap_neighbors, dtype=cp.int32)
                        # For edge costs, use CSR costs for CSR neighbors, default cost for tap neighbors
                        csr_costs = adj_csr.data[start_idx:end_idx] if csr_neighbors else cp.array([], dtype=cp.float32)
                        tap_costs = cp.ones(len(tap_neighbors), dtype=cp.float32) * 1.0  # Default tap edge cost
                        edge_costs = cp.concatenate([csr_costs, tap_costs])
                    elif csr_neighbors:
                        # Only CSR neighbors
                        neighbor_indices = cp.array(csr_neighbors, dtype=cp.int32)
                        edge_costs = adj_csr.data[start_idx:end_idx]
                    elif tap_neighbors:
                        # Only tap neighbors
                        neighbor_indices = cp.array(tap_neighbors, dtype=cp.int32)
                        edge_costs = cp.ones(len(tap_neighbors), dtype=cp.float32) * 1.0  # Default tap edge cost
                    else:
                        continue  # No neighbors at all
                    
                    # Add current congestion costs using proper edge mapping
                    try:
                        # Handle congestion costs for combined neighbor approach
                        if csr_neighbors and tap_neighbors:
                            # Mixed CSR + tap neighbors - map congestion costs separately
                            if hasattr(self.gpu_rrg, 'csr_to_edge_mapping') and csr_neighbors:
                                csr_edge_indices = self.gpu_rrg.csr_to_edge_mapping[start_idx:end_idx]
                                csr_congestion = self.gpu_rrg.pathfinder_state.edge_cost[csr_edge_indices]
                            else:
                                csr_congestion = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                            
                            # Tap edges have no congestion (newly created)
                            tap_congestion = cp.zeros(len(tap_neighbors), dtype=cp.float32)
                            congestion_costs = cp.concatenate([csr_congestion, tap_congestion])
                            
                        elif csr_neighbors:
                            # Only CSR neighbors - use existing logic
                            if hasattr(self.gpu_rrg, 'csr_to_edge_mapping'):
                                original_edge_indices = self.gpu_rrg.csr_to_edge_mapping[start_idx:end_idx]
                                congestion_costs = self.gpu_rrg.pathfinder_state.edge_cost[original_edge_indices]
                            else:
                                congestion_costs = cp.zeros(len(csr_neighbors), dtype=cp.float32)
                                
                        else:
                            # Only tap neighbors - no congestion
                            congestion_costs = cp.zeros(len(tap_neighbors), dtype=cp.float32)
                        
                        # Shape validation and final cost calculation
                        if len(congestion_costs) == len(edge_costs):
                            total_costs = edge_costs + congestion_costs
                        else:
                            logger.warning(f"Congestion cost shape mismatch: {len(congestion_costs)} vs {len(edge_costs)} - using base costs")
                            total_costs = edge_costs
                        
                    except Exception as e:
                        logger.warning(f"Error accessing congestion costs: {e} - using base costs only")
                        total_costs = edge_costs
                    
                    # Calculate new distances
                    current_dist = distances[net_idx, node_idx]
                    new_distances = current_dist + total_costs
                    
                    # Ensure array shapes are compatible
                    if len(new_distances) != len(neighbor_indices):
                        logger.warning(f"Shape mismatch: new_distances {new_distances.shape} vs neighbor_indices {neighbor_indices.shape}")
                        continue
                    
                    # Update for unvisited neighbors with better paths
                    unvisited_mask = ~visited[net_idx, neighbor_indices]
                    better_mask = new_distances < distances[net_idx, neighbor_indices]
                    update_mask = unvisited_mask & better_mask
                    
                    if cp.any(update_mask):
                        update_indices = neighbor_indices[update_mask]
                        distances[net_idx, update_indices] = new_distances[update_mask]
                        parents[net_idx, update_indices] = node_idx
                        frontiers[net_idx, update_indices] = True
        
        # Debug expansion progress
        if step < 5 or step % 100 == 0:  # Log first few steps and every 100th
            logger.info(f"Step {step}: expanded {nodes_expanded} nodes, found {neighbors_found} total neighbors")
            if tap_nodes_expanded > 0:
                logger.info(f"Step {step}: expanded {tap_nodes_expanded} tap nodes, found {tap_neighbors_found} tap neighbors")
            elif step <= 5:
                logger.warning(f"Step {step}: NO TAP NODES EXPANDED - tap nodes may not be in frontiers")
    
    def _calculate_congestion(self, routes: Dict[str, List[int]]) -> cp.ndarray:
        """Calculate edge congestion from all routes"""
        
        # Count usage for each edge
        edge_usage = cp.zeros(self.gpu_rrg.num_edges, dtype=cp.int32)
        
        for net_id, path in routes.items():
            if len(path) > 1:
                # Convert path to edges and increment usage
                for i in range(len(path) - 1):
                    from_node = path[i]
                    to_node = path[i + 1]
                    
                    # Find edge between these nodes
                    edge_idx = self._find_edge_between_nodes(from_node, to_node)
                    if edge_idx >= 0:
                        edge_usage[edge_idx] += 1
        
        return edge_usage
    
    def _update_congestion_costs(self, congestion_map: cp.ndarray, iteration: int):
        """Update edge costs based on congestion using PathFinder formula"""
        
        # PathFinder cost function: base_cost * (1 + pres_fac * (usage - 1))
        state = self.gpu_rrg.pathfinder_state
        
        # Calculate presentation factor for this iteration
        pres_fac = self.pres_fac_base * (self.pres_fac_mult ** iteration)
        
        # Update edge costs
        overused_edges = congestion_map > 1
        congestion_penalty = pres_fac * (congestion_map - 1)
        congestion_penalty = cp.maximum(congestion_penalty, 0)  # No negative costs
        
        # Apply to edge costs (additive congestion costs)
        state.edge_pres_cost[:] = congestion_penalty
        
        # Historical congestion (accumulative)
        state.edge_hist_cost += self.acc_fac * overused_edges.astype(cp.float32)
        
        # Total cost = base + present + historical
        total_congestion = state.edge_pres_cost + state.edge_hist_cost
        state.edge_cost[:] = self.gpu_rrg.edge_base_cost + total_congestion
        
        overused_count = cp.sum(overused_edges)
        logger.info(f"Iteration costs updated: {overused_count} overused edges, pres_fac={pres_fac:.2f}")
    
    def _check_convergence(self, congestion_map: cp.ndarray) -> bool:
        """Check if PathFinder has converged (no congestion)"""
        overused_edges = cp.sum(congestion_map > 1)
        logger.info(f"Convergence check: {overused_edges} overused edges")
        return overused_edges == 0
    
    def _find_edge_between_nodes(self, from_node: int, to_node: int) -> int:
        """Find edge index between two nodes using CSR structure"""
        # Use the appropriate adjacency matrix
        if hasattr(self.gpu_rrg, 'adjacency_csr') and self.gpu_rrg.adjacency_csr is not None:
            adj_csr = self.gpu_rrg.adjacency_csr
        else:
            adj_csr = self.gpu_rrg.adjacency_matrix
        
        if from_node >= len(adj_csr.indptr) - 1:
            return -1
            
        start_idx = int(adj_csr.indptr[from_node])
        end_idx = int(adj_csr.indptr[from_node + 1])
        
        for i in range(start_idx, end_idx):
            if int(adj_csr.indices[i]) == to_node:
                return i  # Return the edge index in the CSR data structure
        
        return -1  # Edge not found
    
    def _reconstruct_path_parallel(self, net_idx: int, source: int, sink: int, parents: cp.ndarray) -> List[int]:
        """Reconstruct path for a single net"""
        path = []
        current = sink
        
        while current != -1 and current != source:
            path.append(int(current))
            current = int(parents[net_idx, current].get() if hasattr(parents, 'get') else parents[net_idx, current])
            
            if len(path) > 10000:  # Prevent infinite loops
                break
        
        if current == source:
            path.append(source)
            path.reverse()
            return path
        else:
            return []  # Path reconstruction failed
    
    def _initialize_congestion_state(self, num_nets: int):
        """Initialize congestion tracking state"""
        state = self.gpu_rrg.pathfinder_state
        
        # Reset congestion costs
        state.edge_pres_cost.fill(0.0)
        state.edge_hist_cost.fill(0.0)
        state.edge_cost[:] = self.gpu_rrg.edge_base_cost.copy()
        
        logger.info(f"Congestion state initialized for {num_nets} nets")