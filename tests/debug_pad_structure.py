#!/usr/bin/env python3
"""
Comprehensive KiCad Pad Structure Analysis
==========================================

This script analyzes the complete KiCad pad structure to understand:
1. Pad types (SMD, through-hole, edge connector, NPTH)
2. Pad shapes (circular, oval, rectangular, trapezoidal, custom)
3. Padstack information (different shapes on different layers)
4. All available pad attributes and properties

Goal: 100% accurate pad rendering with complete understanding of KiCad's pad system.
"""

import logging
import sys
import pprint
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import kipy
    logger.info("✅ Successfully imported kipy")
except ImportError as e:
    logger.error(f"❌ Failed to import kipy: {e}")
    sys.exit(1)

def analyze_pad_complete(pad, pad_index: int) -> Dict:
    """Complete analysis of a single pad with all attributes"""
    analysis = {
        'index': pad_index,
        'basic_info': {},
        'geometry': {},
        'type_info': {},
        'shape_info': {},
        'padstack_info': {},
        'layer_info': {},
        'drill_info': {},
        'raw_attributes': {}
    }
    
    # === BASIC INFORMATION ===
    try:
        analysis['basic_info'] = {
            'position': {
                'x_mm': getattr(pad.position, 'x', 0) / 1e6 if hasattr(pad, 'position') else 0,
                'y_mm': getattr(pad.position, 'y', 0) / 1e6 if hasattr(pad, 'position') else 0,
                'raw_x': getattr(pad.position, 'x', 0) if hasattr(pad, 'position') else 0,
                'raw_y': getattr(pad.position, 'y', 0) if hasattr(pad, 'position') else 0,
            },
            'net': getattr(pad, 'net', ''),
            'number': getattr(pad, 'number', ''),
            'name': getattr(pad, 'name', ''),
        }
    except Exception as e:
        logger.warning(f"Error getting basic info: {e}")
    
    # === PAD TYPE ANALYSIS ===
    try:
        # Check for pad type/attribute
        pad_type = getattr(pad, 'type', None)
        pad_attribute = getattr(pad, 'attribute', None)
        pad_attrib = getattr(pad, 'attrib', None)
        
        analysis['type_info'] = {
            'type': pad_type,
            'attribute': pad_attribute, 
            'attrib': pad_attrib,
            'type_raw': str(pad_type) if pad_type is not None else None,
            'attribute_raw': str(pad_attribute) if pad_attribute is not None else None,
            'attrib_raw': str(pad_attrib) if pad_attrib is not None else None,
        }
    except Exception as e:
        logger.warning(f"Error getting type info: {e}")
    
    # === GEOMETRY ANALYSIS ===
    try:
        size = getattr(pad, 'size', None)
        analysis['geometry'] = {
            'size': {
                'x_mm': getattr(size, 'x', 0) / 1e6 if size else 0,
                'y_mm': getattr(size, 'y', 0) / 1e6 if size else 0,
                'raw_x': getattr(size, 'x', 0) if size else 0,
                'raw_y': getattr(size, 'y', 0) if size else 0,
            } if size else None,
            'orientation': getattr(pad, 'orientation', 0),
            'offset': getattr(pad, 'offset', None),
        }
    except Exception as e:
        logger.warning(f"Error getting geometry: {e}")
    
    # === SHAPE ANALYSIS ===
    try:
        shape = getattr(pad, 'shape', None)
        analysis['shape_info'] = {
            'shape': shape,
            'shape_raw': str(shape) if shape is not None else None,
            'shape_type': type(shape).__name__ if shape is not None else None,
        }
    except Exception as e:
        logger.warning(f"Error getting shape info: {e}")
    
    # === PADSTACK ANALYSIS ===
    try:
        padstack = getattr(pad, 'padstack', None)
        if padstack:
            analysis['padstack_info'] = {
                'has_padstack': True,
                'copper_layers': [],
                'mode': getattr(padstack, 'mode', None),
                'type': getattr(padstack, 'type', None),
            }
            
            # Analyze copper layers
            copper_layers = getattr(padstack, 'copper_layers', [])
            for i, layer in enumerate(copper_layers):
                layer_info = {
                    'index': i,
                    'shape': getattr(layer, 'shape', None),
                    'shape_raw': str(getattr(layer, 'shape', None)),
                    'size': {
                        'x': getattr(getattr(layer, 'size', None), 'x', None),
                        'y': getattr(getattr(layer, 'size', None), 'y', None),
                    } if hasattr(layer, 'size') else None,
                }
                analysis['padstack_info']['copper_layers'].append(layer_info)
        else:
            analysis['padstack_info'] = {'has_padstack': False}
    except Exception as e:
        logger.warning(f"Error getting padstack info: {e}")
        analysis['padstack_info'] = {'error': str(e)}
    
    # === DRILL ANALYSIS ===
    try:
        drill_size = getattr(pad, 'drill_size', None)
        analysis['drill_info'] = {
            'drill_size': {
                'x_mm': getattr(drill_size, 'x', 0) / 1e6 if drill_size else 0,
                'y_mm': getattr(drill_size, 'y', 0) / 1e6 if drill_size else 0,
                'raw_x': getattr(drill_size, 'x', 0) if drill_size else 0,
                'raw_y': getattr(drill_size, 'y', 0) if drill_size else 0,
            } if drill_size else None,
            'drill_shape': getattr(pad, 'drill_shape', None),
        }
    except Exception as e:
        logger.warning(f"Error getting drill info: {e}")
    
    # === LAYER ANALYSIS ===
    try:
        layers = getattr(pad, 'layers', [])
        analysis['layer_info'] = {
            'layers': [str(layer) for layer in layers],
            'layer_count': len(layers),
        }
    except Exception as e:
        logger.warning(f"Error getting layer info: {e}")
    
    # === RAW ATTRIBUTE DUMP ===
    try:
        # Get all attributes of the pad object
        raw_attrs = {}
        for attr_name in dir(pad):
            if not attr_name.startswith('_'):  # Skip private attributes
                try:
                    attr_value = getattr(pad, attr_name)
                    if not callable(attr_value):  # Skip methods
                        raw_attrs[attr_name] = {
                            'value': str(attr_value),
                            'type': type(attr_value).__name__,
                        }
                except Exception:
                    raw_attrs[attr_name] = {'error': 'Could not access'}
        
        analysis['raw_attributes'] = raw_attrs
    except Exception as e:
        logger.warning(f"Error getting raw attributes: {e}")
    
    return analysis

def main():
    """Analyze KiCad pad structure comprehensively"""
    logger.info("🔍 Starting comprehensive KiCad pad structure analysis...")
    
    try:
        # Connect to KiCad using same method as kicad_interface.py
        import os
        
        api_socket = os.environ.get('KICAD_API_SOCKET')
        api_token = os.environ.get('KICAD_API_TOKEN')
        timeout_ms = 25000
        if api_socket or api_token:
            client = kipy.KiCad(socket_path=api_socket, kicad_token=api_token, timeout_ms=timeout_ms)
        else:
            client = kipy.KiCad(timeout_ms=timeout_ms)
        
        # Get board - try different methods
        try:
            board = client.get_board()
        except Exception as e1:
            logger.warning(f"get_board failed: {e1}")
            try:
                docs = client.get_open_documents()
                if docs and len(docs) > 0:
                    board = docs[0]
                else:
                    raise Exception("No open documents found")
            except Exception as e2:
                logger.warning(f"get_open_documents failed: {e2}")
                board = client.board
        logger.info("✅ Connected to KiCad IPC API and retrieved board")
        
        # Get all footprints using the working method
        try:
            footprints = board.get_footprints()
            logger.info(f"Found {len(footprints)} footprints via get_footprints()")
        except Exception as e:
            logger.warning(f"get_footprints() failed: {e}")
            try:
                footprints = getattr(board, 'footprints', [])
                logger.info(f"Found {len(footprints)} footprints via .footprints")
            except Exception as e2:
                logger.error(f"All footprint access methods failed: {e2}")
                footprints = []
        
        # Get pads directly from board using the working method
        try:
            all_pads = board.get_pads()
            logger.info(f"Found {len(all_pads)} pads via board.get_pads()")
        except Exception as e:
            logger.warning(f"board.get_pads() failed: {e}")
            # Fallback: try getting from footprints
            all_pads = []
            for footprint in footprints:
                try:
                    pads = getattr(footprint, 'pads', [])
                    if hasattr(footprint, 'get_pads'):
                        pads = footprint.get_pads()
                    all_pads.extend(pads)
                    logger.info(f"Footprint '{getattr(footprint, 'reference', 'Unknown')}' has {len(pads)} pads")
                except Exception as pad_e:
                    logger.warning(f"Error getting pads from footprint: {pad_e}")
        
        logger.info(f"Found {len(all_pads)} total pads")
        
        # Analyze first 10 pads in detail
        logger.info("\n" + "="*80)
        logger.info("DETAILED PAD ANALYSIS")
        logger.info("="*80)
        
        for i, pad in enumerate(all_pads[:10]):
            logger.info(f"\n📍 PAD {i} ANALYSIS:")
            logger.info("-" * 50)
            
            analysis = analyze_pad_complete(pad, i)
            
            # Print structured analysis
            for section, data in analysis.items():
                if data and section != 'raw_attributes':  # Skip empty sections and raw dump for now
                    logger.info(f"\n{section.upper()}:")
                    if isinstance(data, dict):
                        for key, value in data.items():
                            logger.info(f"  {key}: {value}")
                    else:
                        logger.info(f"  {data}")
        
        # Analyze patterns across all pads
        logger.info("\n" + "="*80)
        logger.info("PAD PATTERN ANALYSIS")
        logger.info("="*80)
        
        pad_types = {}
        pad_shapes = {}
        pad_sizes = {}
        
        for i, pad in enumerate(all_pads):
            # Collect pad type patterns
            pad_type = getattr(pad, 'type', None)
            pad_attribute = getattr(pad, 'attribute', None)
            
            type_key = f"type={pad_type}, attr={pad_attribute}"
            pad_types[type_key] = pad_types.get(type_key, 0) + 1
            
            # Collect shape patterns  
            shape = getattr(pad, 'shape', None)
            padstack = getattr(pad, 'padstack', None)
            if padstack and hasattr(padstack, 'copper_layers') and len(padstack.copper_layers) > 0:
                shape = padstack.copper_layers[0].shape
            
            pad_shapes[str(shape)] = pad_shapes.get(str(shape), 0) + 1
            
            # Collect size patterns
            size = getattr(pad, 'size', None)
            if size:
                size_key = f"{getattr(size, 'x', 0)/1e6:.3f}x{getattr(size, 'y', 0)/1e6:.3f}mm"
                pad_sizes[size_key] = pad_sizes.get(size_key, 0) + 1
        
        logger.info("\nPAD TYPE DISTRIBUTION:")
        for pad_type, count in sorted(pad_types.items()):
            logger.info(f"  {pad_type}: {count} pads")
        
        logger.info("\nPAD SHAPE DISTRIBUTION:")
        for shape, count in sorted(pad_shapes.items()):
            logger.info(f"  Shape {shape}: {count} pads")
        
        logger.info("\nPAD SIZE DISTRIBUTION (top 10):")
        sorted_sizes = sorted(pad_sizes.items(), key=lambda x: x[1], reverse=True)
        for size, count in sorted_sizes[:10]:
            logger.info(f"  {size}: {count} pads")
        
        logger.info("\n✅ Pad structure analysis complete!")
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
