#!/usr/bin/env python3
"""
Route Buffer - Holds routing solution before committing to KiCad
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PendingTrack:
    """A track waiting to be committed to KiCad"""
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    layer: str
    width: float
    net_name: str

@dataclass
class PendingVia:
    """A via waiting to be committed to KiCad"""
    x: float
    y: float
    size: float
    drill: float
    from_layer: str
    to_layer: str
    net_name: str

class RouteBuffer:
    """Holds routing solution before committing to KiCad"""
    
    def __init__(self):
        self.pending_tracks: List[PendingTrack] = []
        self.pending_vias: List[PendingVia] = []
        self.routed_nets: set = set()
        self.total_length: float = 0.0
        self.via_count: int = 0
    
    def add_track(self, start_x: float, start_y: float, end_x: float, end_y: float,
                  layer: str, width: float, net_name: str):
        """Add a track to the routing solution"""
        track = PendingTrack(start_x, start_y, end_x, end_y, layer, width, net_name)
        self.pending_tracks.append(track)
        self.routed_nets.add(net_name)
        
        # Calculate length
        import math
        length = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        self.total_length += length
        
        logger.debug(f"Added track for {net_name}: ({start_x:.2f},{start_y:.2f}) → ({end_x:.2f},{end_y:.2f})")
    
    def add_via(self, x: float, y: float, size: float, drill: float,
                from_layer: str, to_layer: str, net_name: str):
        """Add a via to the routing solution"""
        via = PendingVia(x, y, size, drill, from_layer, to_layer, net_name)
        self.pending_vias.append(via)
        self.via_count += 1
        
        logger.debug(f"Added via for {net_name}: ({x:.2f},{y:.2f}) {from_layer}→{to_layer}")
    
    def get_stats(self) -> Dict:
        """Get routing statistics"""
        return {
            'nets_routed': len(self.routed_nets),
            'tracks': len(self.pending_tracks),
            'vias': len(self.pending_vias),
            'total_length_mm': round(self.total_length, 2),
            'via_count': self.via_count
        }
    
    def clear(self):
        """Clear all pending routes"""
        self.pending_tracks.clear()
        self.pending_vias.clear()
        self.routed_nets.clear()
        self.total_length = 0.0
        self.via_count = 0
    
    def commit_to_kicad(self, kicad_interface) -> bool:
        """Commit all pending routes to KiCad"""
        try:
            logger.info(f"Committing {len(self.pending_tracks)} tracks and {len(self.pending_vias)} vias to KiCad...")
            
            success_count = 0
            
            # Create all tracks
            for track in self.pending_tracks:
                if kicad_interface.create_track(
                    track.start_x, track.start_y, track.end_x, track.end_y,
                    track.layer, track.width, track.net_name
                ):
                    success_count += 1
            
            # Create all vias
            for via in self.pending_vias:
                if kicad_interface.create_via(
                    via.x, via.y, via.size, via.drill,
                    via.from_layer, via.to_layer, via.net_name
                ):
                    success_count += 1
            
            total_items = len(self.pending_tracks) + len(self.pending_vias)
            
            if success_count == total_items:
                logger.info(f"✅ Successfully committed {success_count}/{total_items} items to KiCad")
                kicad_interface.refresh_board()  # Refresh KiCad display
                return True
            else:
                logger.warning(f"⚠️ Only {success_count}/{total_items} items committed successfully")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error committing routes to KiCad: {e}")
            return False
    
    def export_to_board_data(self) -> Dict:
        """Export routes for visualization in OrthoRoute GUI"""
        tracks = []
        vias = []
        
        for track in self.pending_tracks:
            tracks.append({
                'start_x': track.start_x,
                'start_y': track.start_y,
                'end_x': track.end_x,
                'end_y': track.end_y,
                'layer': track.layer,
                'width': track.width,
                'net': track.net_name
            })
        
        for via in self.pending_vias:
            vias.append({
                'x': via.x,
                'y': via.y,
                'via_diameter': via.size,
                'drill_diameter': via.drill,
                'net': via.net_name
            })
        
        return {
            'tracks': tracks,
            'vias': vias,
            'stats': self.get_stats()
        }


class RoutingSession:
    """Manages a complete routing session with preview and commit"""
    
    def __init__(self, kicad_interface, board_data):
        self.kicad_interface = kicad_interface
        self.board_data = board_data
        self.route_buffer = RouteBuffer()
        self.original_tracks = board_data.get('tracks', []).copy()
        self.original_vias = board_data.get('vias', []).copy()
    
    def route_net(self, net_name: str, algorithm: str = 'auto') -> bool:
        """Route a single net using specified algorithm"""
        # This is where we'd call the actual routing algorithms
        # from src/unused/ directory
        logger.info(f"Routing net '{net_name}' using {algorithm} algorithm...")
        
        # TODO: Integrate with actual routing algorithms
        # - frontier_reduction_router.py
        # - lees_routing_adapter.py  
        # - gpu_routing_engine.py
        
        return True
    
    def route_all(self, algorithm: str = 'auto') -> bool:
        """Route all unrouted nets"""
        unrouted_nets = [net for net in self.board_data.get('nets', []) 
                        if not net.get('routed', False)]
        
        logger.info(f"Starting batch routing of {len(unrouted_nets)} nets...")
        
        success_count = 0
        for net in unrouted_nets:
            if self.route_net(net['name'], algorithm):
                success_count += 1
        
        logger.info(f"Routing complete: {success_count}/{len(unrouted_nets)} nets routed")
        return success_count > 0
    
    def preview_solution(self) -> Dict:
        """Get routing solution for preview (without committing)"""
        # Update board data with pending routes for visualization
        preview_data = self.board_data.copy()
        route_data = self.route_buffer.export_to_board_data()
        
        # Merge original and new routes
        preview_data['tracks'] = self.original_tracks + route_data['tracks']
        preview_data['vias'] = self.original_vias + route_data['vias']
        
        return preview_data
    
    def commit_solution(self) -> bool:
        """Commit the routing solution to KiCad"""
        return self.route_buffer.commit_to_kicad(self.kicad_interface)
    
    def rollback(self):
        """Discard current routing solution"""
        self.route_buffer.clear()
        logger.info("Routing solution discarded")
    
    def get_solution_stats(self) -> Dict:
        """Get statistics about current solution"""
        return self.route_buffer.get_stats()


if __name__ == "__main__":
    # Example usage
    buffer = RouteBuffer()
    
    # Add some example routes
    buffer.add_track(10, 10, 20, 10, 'F.Cu', 0.2, 'VCC')
    buffer.add_via(20, 10, 0.6, 0.3, 'F.Cu', 'B.Cu', 'VCC')
    buffer.add_track(20, 10, 30, 20, 'B.Cu', 0.2, 'VCC')
    
    print("Route Buffer Stats:", buffer.get_stats())
