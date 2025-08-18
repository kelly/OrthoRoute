"""
Virtual Copper Pour Generator

Creates a virtual routable area polygon for each layer, independent of KiCad's copper pours.
This polygon defines where traces can be routed based on:
- Board outline (maximum area)
- Pad clearances (keepout zones)
- Existing tracks/vias (obstacles)
- Board features (holes, cutouts)

The virtual copper pour exists only in our GPU routing model.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..core.board_interface import BoardInterface
from ..core.drc_rules import DRCRules

logger = logging.getLogger(__name__)


"""
Virtual Copper Pour Generator

Creates a virtual routable area grid for each layer, independent of KiCad's copper pours.
This defines where traces can be routed based on:
- Board outline (maximum area)
- Pad clearances (keepout zones)
- Existing tracks/vias (obstacles)

The virtual copper pour exists only in our GPU routing model.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import sys
import os

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

logger = logging.getLogger(__name__)


class VirtualCopperGenerator:
    """Generates virtual copper pour grids for PCB routing"""
    
    def __init__(self, board_interface, drc_rules, grid_config):
        self.board_interface = board_interface
        self.drc_rules = drc_rules
        self.grid_config = grid_config
        
        # Cache for generated grids
        self._virtual_copper_cache: Dict[str, np.ndarray] = {}
        
        logger.info("🔧 Virtual Copper Generator initialized")
    
    def generate_virtual_copper_grid(self, layer: str) -> np.ndarray:
        """
        Generate virtual copper pour grid for a specific layer
        
        Returns:
            np.ndarray: Boolean grid where True = routable area, False = obstacle
        """
        if layer in self._virtual_copper_cache:
            return self._virtual_copper_cache[layer]
        
        logger.info(f"🌟 Generating virtual copper pour for layer {layer}")
        
        # Initialize grid as all routable (True = free space)
        height, width = self.grid_config.height, self.grid_config.width
        virtual_copper_grid = np.ones((height, width), dtype=bool)
        
        # Step 1: Apply board outline constraints
        virtual_copper_grid = self._apply_board_outline_constraints(virtual_copper_grid)
        
        # Step 2: Apply pad keepout zones
        virtual_copper_grid = self._apply_pad_keepouts(virtual_copper_grid, layer)
        
        # Step 3: Apply existing track obstacles
        virtual_copper_grid = self._apply_track_obstacles(virtual_copper_grid, layer)
        
        # Step 4: Apply via obstacles
        virtual_copper_grid = self._apply_via_obstacles(virtual_copper_grid, layer)
        
        # Step 5: Apply board feature obstacles
        virtual_copper_grid = self._apply_board_feature_obstacles(virtual_copper_grid)
        
        # Cache the result
        self._virtual_copper_cache[layer] = virtual_copper_grid
        
        routable_cells = np.sum(virtual_copper_grid)
        total_cells = height * width
        percentage = (routable_cells / total_cells) * 100
        
        logger.info(f"✅ Virtual copper pour for {layer}: {routable_cells}/{total_cells} cells routable ({percentage:.1f}%)")
        
        return virtual_copper_grid
    
    def _apply_board_outline_constraints(self, grid: np.ndarray) -> np.ndarray:
        """Apply board outline as routing boundary"""
        try:
            # Get board outline from KiCad
            outline_points = self.board_interface.get_board_outline()
            
            if not outline_points or len(outline_points) < 3:
                logger.warning("⚠️ No valid board outline found, using full grid")
                return grid  # Keep full grid as routable
            
            # For each grid cell, check if it's inside the board outline
            height, width = grid.shape
            
            for gy in range(height):
                for gx in range(width):
                    # Convert grid coordinates to world coordinates
                    world_x, world_y = self.grid_config.grid_to_world(gx, gy)
                    
                    # Simple point-in-polygon test using winding number
                    if not self._point_in_polygon(world_x, world_y, outline_points):
                        grid[gy, gx] = False  # Outside board = not routable
            
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying board outline constraints: {e}")
            return grid  # Return original grid as fallback
    
    def _apply_pad_keepouts(self, grid: np.ndarray, layer: str) -> np.ndarray:
        """Apply keepout zones around all pads"""
        try:
            pads = self.board_interface.get_all_pads()
            height, width = grid.shape
            
            pad_count = 0
            for pad in pads:
                # Check if pad affects this layer
                if not self._pad_affects_layer(pad, layer):
                    continue
                
                # Get pad position and size
                pad_x, pad_y = self.board_interface.get_pad_center(pad)
                pad_size = self.board_interface.get_pad_size(pad)
                
                # Calculate keepout radius (pad size + clearance)
                pad_radius = max(pad_size.get('width', 0), pad_size.get('height', 0)) / 2.0
                keepout_radius = pad_radius + self.drc_rules.min_trace_spacing
                
                # Convert to grid coordinates
                pad_gx, pad_gy = self.grid_config.world_to_grid(pad_x, pad_y)
                keepout_radius_grid = keepout_radius / self.grid_config.resolution
                
                # Mark cells within keepout radius as obstacles
                for gy in range(max(0, int(pad_gy - keepout_radius_grid)), 
                               min(height, int(pad_gy + keepout_radius_grid) + 1)):
                    for gx in range(max(0, int(pad_gx - keepout_radius_grid)), 
                                   min(width, int(pad_gx + keepout_radius_grid) + 1)):
                        
                        # Calculate distance from pad center
                        distance = np.sqrt((gx - pad_gx)**2 + (gy - pad_gy)**2) * self.grid_config.resolution
                        
                        if distance <= keepout_radius:
                            grid[gy, gx] = False  # Mark as obstacle
                
                pad_count += 1
            
            logger.debug(f"📦 Applied keepouts for {pad_count} pads on {layer}")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying pad keepouts: {e}")
            return grid
    
    def _apply_track_obstacles(self, grid: np.ndarray, layer: str) -> np.ndarray:
        """Apply obstacles for existing tracks"""
        try:
            tracks = self.board_interface.get_all_tracks()
            layer_id = self.board_interface.get_layer_id(layer)
            
            track_count = 0
            for track in tracks:
                track_geom = self.board_interface.get_track_geometry(track)
                
                # Only process tracks on this layer
                if track_geom['layer'] != layer_id:
                    continue
                
                # Mark track area as obstacle using line drawing
                self._mark_line_obstacle(
                    grid,
                    track_geom['start_x'], track_geom['start_y'],
                    track_geom['end_x'], track_geom['end_y'],
                    track_geom['width']
                )
                
                track_count += 1
            
            logger.debug(f"🛤️ Applied {track_count} track obstacles on {layer}")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying track obstacles: {e}")
            return grid
    
    def _apply_via_obstacles(self, grid: np.ndarray, layer: str) -> np.ndarray:
        """Apply obstacles for vias"""
        try:
            vias = self.board_interface.get_all_vias()
            layer_id = self.board_interface.get_layer_id(layer)
            
            via_count = 0
            for via in vias:
                via_geom = self.board_interface.get_via_geometry(via)
                
                # Check if via affects this layer
                if not self._via_affects_layer(via_geom, layer_id):
                    continue
                
                # Mark circular area around via as obstacle
                via_gx, via_gy = self.grid_config.world_to_grid(via_geom['x'], via_geom['y'])
                via_radius = (via_geom['drill_diameter'] / 2.0 + self.drc_rules.min_via_spacing)
                via_radius_grid = via_radius / self.grid_config.resolution
                
                height, width = grid.shape
                for gy in range(max(0, int(via_gy - via_radius_grid)), 
                               min(height, int(via_gy + via_radius_grid) + 1)):
                    for gx in range(max(0, int(via_gx - via_radius_grid)), 
                                   min(width, int(via_gx + via_radius_grid) + 1)):
                        
                        distance = np.sqrt((gx - via_gx)**2 + (gy - via_gy)**2) * self.grid_config.resolution
                        if distance <= via_radius:
                            grid[gy, gx] = False
                
                via_count += 1
            
            logger.debug(f"🔘 Applied {via_count} via obstacles on {layer}")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying via obstacles: {e}")
            return grid
    
    def _apply_board_feature_obstacles(self, grid: np.ndarray) -> np.ndarray:
        """Apply obstacles for board features like holes"""
        try:
            holes = self.board_interface.get_board_holes()
            
            hole_count = 0
            for hole in holes:
                hole_gx, hole_gy = self.grid_config.world_to_grid(hole['x'], hole['y'])
                hole_radius = (hole['diameter'] / 2.0 + self.drc_rules.min_hole_clearance)
                hole_radius_grid = hole_radius / self.grid_config.resolution
                
                height, width = grid.shape
                for gy in range(max(0, int(hole_gy - hole_radius_grid)), 
                               min(height, int(hole_gy + hole_radius_grid) + 1)):
                    for gx in range(max(0, int(hole_gx - hole_radius_grid)), 
                                   min(width, int(hole_gx + hole_radius_grid) + 1)):
                        
                        distance = np.sqrt((gx - hole_gx)**2 + (gy - hole_gy)**2) * self.grid_config.resolution
                        if distance <= hole_radius:
                            grid[gy, gx] = False
                
                hole_count += 1
            
            logger.debug(f"🕳️ Applied {hole_count} board feature obstacles")
            return grid
            
        except Exception as e:
            logger.error(f"❌ Error applying board feature obstacles: {e}")
            return grid
    
    def _mark_line_obstacle(self, grid: np.ndarray, start_x: float, start_y: float, 
                           end_x: float, end_y: float, width: float):
        """Mark a line as an obstacle in the grid"""
        # Convert to grid coordinates
        start_gx, start_gy = self.grid_config.world_to_grid(start_x, start_y)
        end_gx, end_gy = self.grid_config.world_to_grid(end_x, end_y)
        width_grid = width / self.grid_config.resolution
        
        # Use Bresenham's line algorithm with width
        dx = abs(end_gx - start_gx)
        dy = abs(end_gy - start_gy)
        sx = 1 if start_gx < end_gx else -1
        sy = 1 if start_gy < end_gy else -1
        err = dx - dy
        
        x, y = start_gx, start_gy
        height, width_cells = grid.shape
        
        while True:
            # Mark circle around current point
            for dy_offset in range(int(-width_grid), int(width_grid) + 1):
                for dx_offset in range(int(-width_grid), int(width_grid) + 1):
                    if dx_offset**2 + dy_offset**2 <= width_grid**2:
                        px, py = int(x + dx_offset), int(y + dy_offset)
                        if 0 <= px < width_cells and 0 <= py < height:
                            grid[py, px] = False
            
            if x == end_gx and y == end_gy:
                break
                
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    def _point_in_polygon(self, x: float, y: float, polygon_points: List[Tuple[float, float]]) -> bool:
        """Test if point is inside polygon using ray casting"""
        n = len(polygon_points)
        inside = False
        
        p1x, p1y = polygon_points[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon_points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _pad_affects_layer(self, pad, layer: str) -> bool:
        """Check if a pad affects routing on the given layer"""
        try:
            pad_layers = self.board_interface.get_pad_layers(pad)
            layer_id = self.board_interface.get_layer_id(layer)
            
            # Pad affects layer if it's on that layer or is through-hole
            return layer_id in pad_layers or self.board_interface.is_pad_through_hole(pad)
            
        except Exception:
            return True  # Conservative: assume it affects the layer
    
    def _via_affects_layer(self, via_geom: dict, layer_id: int) -> bool:
        """Check if a via affects routing on the given layer"""
        try:
            start_layer = via_geom.get('start_layer', 0)
            end_layer = via_geom.get('end_layer', 31)
            
            # Via affects layer if it spans through it
            return start_layer <= layer_id <= end_layer
            
        except Exception:
            return True  # Conservative: assume it affects the layer
    
    def clear_cache(self):
        """Clear the virtual copper cache"""
        self._virtual_copper_cache.clear()
        logger.debug("🧹 Virtual copper cache cleared")
