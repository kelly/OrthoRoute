"""
OrthoRoute API Bridge
Provides compatibility layer between SWIG and IPC APIs
"""

import sys

# API Detection and Import
SWIG_AVAILABLE = False
IPC_AVAILABLE = False

try:
    import pcbnew
    SWIG_AVAILABLE = True
    print("✅ SWIG API (pcbnew) available")
except ImportError:
    print("❌ SWIG API not available")

try:
    from kicad.pcbnew.board import Board as IPCBoard
    from kicad.pcbnew.module import Module as IPCModule
    IPC_AVAILABLE = True
    print("✅ IPC API (kicad-python) available")
except ImportError:
    print("⚠️ IPC API not available")

if not SWIG_AVAILABLE and not IPC_AVAILABLE:
    raise ImportError("Neither SWIG nor IPC API available!")

class APIBridge:
    """Bridge class for KiCad API compatibility"""
    
    def __init__(self):
        self.api_type = None
        self.board = None
        
        if IPC_AVAILABLE:
            self.api_type = "IPC"
            print("🔄 Using IPC API (kicad-python)")
        elif SWIG_AVAILABLE:
            self.api_type = "SWIG"
            print("🔄 Using SWIG API (deprecated)")
        else:
            raise RuntimeError("No compatible API available")

    def get_board(self):
        """Get board using available API"""
        if self.api_type == "IPC":
            try:
                # Get SWIG board first, then wrap with IPC
                swig_board = pcbnew.GetBoard()
                self.board = IPCBoard.wrap(swig_board)
                return self.board
            except Exception as e:
                print(f"❌ IPC board access failed: {e}")
                # Fallback to SWIG
                self.api_type = "SWIG"
                
        if self.api_type == "SWIG":
            self.board = pcbnew.GetBoard()
            return self.board
            
        return None

    def extract_board_data(self, board=None):
        """Extract board data using available API"""
        if board is None:
            board = self.get_board()
            
        if not board:
            return None
            
        if self.api_type == "IPC":
            return self._extract_board_data_ipc(board)
        else:
            return self._extract_board_data_swig(board)

    def _extract_board_data_ipc(self, ipc_board):
        """Extract board data using IPC API"""
        try:
            # Get native SWIG board for some operations
            swig_board = ipc_board.native_obj
            
            # Board bounds
            bounds = swig_board.GetBoardEdgesBoundingBox()
            board_data = {
                'bounds': {
                    'width_nm': bounds.GetWidth(),
                    'height_nm': bounds.GetHeight(),
                    'layers': swig_board.GetCopperLayerCount()
                },
                'nets': [],
                'obstacles': {}
            }
            
            # Extract nets using hybrid approach
            netcodes = swig_board.GetNetsByNetcode()
            
            for netcode, net in netcodes.items():
                if netcode == 0:  # Skip unconnected
                    continue
                    
                net_name = net.GetNetname()
                if not net_name:
                    continue
                
                # Find pads for this net
                net_pads = []
                for module in ipc_board.modules:
                    # Access native module for pad iteration
                    native_module = module.native_obj
                    for pad in native_module.Pads():
                        pad_net = pad.GetNet()
                        if pad_net.GetNetCode() == netcode:
                            pos = pad.GetPosition()
                            net_pads.append({
                                'x': pos.x,
                                'y': pos.y,
                                'layer': 0,  # Could be extracted from pad layers
                                'pad_name': pad.GetName()
                            })
                
                if len(net_pads) >= 2:
                    board_data['nets'].append({
                        'id': netcode,
                        'name': net_name,
                        'pins': net_pads,
                        'width_nm': 200000,  # Default
                        'kicad_net': net
                    })
            
            return board_data
            
        except Exception as e:
            print(f"❌ IPC board data extraction failed: {e}")
            # Fallback to SWIG
            return self._extract_board_data_swig(ipc_board.native_obj)

    def _extract_board_data_swig(self, swig_board):
        """Extract board data using SWIG API"""
        try:
            # Board bounds
            bounds = swig_board.GetBoardEdgesBoundingBox()
            board_data = {
                'bounds': {
                    'width_nm': bounds.GetWidth(),
                    'height_nm': bounds.GetHeight(),
                    'layers': swig_board.GetCopperLayerCount()
                },
                'nets': [],
                'obstacles': {}
            }
            
            # Extract nets
            netcodes = swig_board.GetNetsByNetcode()
            
            for netcode, net in netcodes.items():
                if netcode == 0:  # Skip unconnected
                    continue
                    
                net_name = net.GetNetname()
                if not net_name:
                    continue
                
                # Find pads for this net using corrected logic
                net_pads = []
                for footprint in swig_board.GetFootprints():
                    for pad in footprint.Pads():
                        pad_net = pad.GetNet()
                        if pad_net.GetNetCode() == netcode:
                            pos = pad.GetPosition()
                            net_pads.append({
                                'x': pos.x,
                                'y': pos.y,
                                'layer': 0,  # Could be extracted from pad layers
                                'pad_name': pad.GetName()
                            })
                
                if len(net_pads) >= 2:
                    board_data['nets'].append({
                        'id': netcode,
                        'name': net_name,
                        'pins': net_pads,
                        'width_nm': 200000,  # Default
                        'kicad_net': net
                    })
            
            return board_data
            
        except Exception as e:
            print(f"❌ SWIG board data extraction failed: {e}")
            return None

    def create_track(self, start_pos, end_pos, layer=0, width=200000, net=None, board=None):
        """Create track using available API"""
        if board is None:
            board = self.board
            
        if self.api_type == "IPC":
            return self._create_track_ipc(start_pos, end_pos, layer, width, net, board)
        else:
            return self._create_track_swig(start_pos, end_pos, layer, width, net, board)

    def _create_track_ipc(self, start_pos, end_pos, layer, width, net, ipc_board):
        """Create track using IPC API"""
        try:
            # Use IPC API high-level track creation
            coords = [
                (start_pos['x'] / 1e6, start_pos['y'] / 1e6),  # Convert to mm
                (end_pos['x'] / 1e6, end_pos['y'] / 1e6)
            ]
            
            # IPC API uses mm and layer names
            layer_name = 'F.Cu' if layer == 0 else 'B.Cu'
            width_mm = width / 1e6
            
            track = ipc_board.add_track(coords, layer=layer_name, width=width_mm)
            return track
            
        except Exception as e:
            print(f"❌ IPC track creation failed: {e}")
            # Fallback to SWIG
            return self._create_track_swig(start_pos, end_pos, layer, width, net, ipc_board.native_obj)

    def _create_track_swig(self, start_pos, end_pos, layer, width, net, swig_board):
        """Create track using SWIG API"""
        try:
            # Create track using SWIG API
            track = pcbnew.PCB_TRACK(swig_board)
            
            # Set positions
            track.SetStart(pcbnew.VECTOR2I(int(start_pos['x']), int(start_pos['y'])))
            track.SetEnd(pcbnew.VECTOR2I(int(end_pos['x']), int(end_pos['y'])))
            
            # Set layer
            layer_id = pcbnew.F_Cu if layer == 0 else pcbnew.B_Cu
            track.SetLayer(layer_id)
            
            # Set width
            track.SetWidth(width)
            
            # Set net
            if net:
                track.SetNet(net)
            
            # Add to board
            swig_board.Add(track)
            
            return track
            
        except Exception as e:
            print(f"❌ SWIG track creation failed: {e}")
            return None

    def create_via(self, position, layer_pair=(0, 1), size=None, drill=None, net=None, board=None):
        """Create via using available API"""
        if board is None:
            board = self.board
            
        if self.api_type == "IPC":
            return self._create_via_ipc(position, layer_pair, size, drill, net, board)
        else:
            return self._create_via_swig(position, layer_pair, size, drill, net, board)

    def _create_via_ipc(self, position, layer_pair, size, drill, net, ipc_board):
        """Create via using IPC API"""
        try:
            # Convert position to mm
            coord = (position['x'] / 1e6, position['y'] / 1e6)
            
            # Convert layer numbers to layer names
            layer_names = ['F.Cu', 'B.Cu', 'In1.Cu', 'In2.Cu']
            layer_pair_names = (layer_names[layer_pair[0]], layer_names[layer_pair[1]])
            
            # Convert sizes to mm
            size_mm = (size or 0.8) / 1e6 if isinstance(size, (int, float)) else 0.8
            drill_mm = (drill or 0.4) / 1e6 if isinstance(drill, (int, float)) else 0.4
            
            via = ipc_board.add_via(coord, layer_pair_names, size=size_mm, drill=drill_mm)
            return via
            
        except Exception as e:
            print(f"❌ IPC via creation failed: {e}")
            # Fallback to SWIG
            return self._create_via_swig(position, layer_pair, size, drill, net, ipc_board.native_obj)

    def _create_via_swig(self, position, layer_pair, size, drill, net, swig_board):
        """Create via using SWIG API"""
        try:
            # Create via using SWIG API
            via = pcbnew.PCB_VIA(swig_board)
            
            # Set position
            via.SetPosition(pcbnew.VECTOR2I(int(position['x']), int(position['y'])))
            
            # Set via type
            via.SetViaType(pcbnew.VIATYPE_THROUGH)
            
            # Set size and drill
            if size:
                via.SetWidth(size)
            if drill:
                via.SetDrill(drill)
            
            # Set net
            if net:
                via.SetNet(net)
            
            # Add to board
            swig_board.Add(via)
            
            return via
            
        except Exception as e:
            print(f"❌ SWIG via creation failed: {e}")
            return None

    def get_api_info(self):
        """Get information about available APIs"""
        return {
            'current_api': self.api_type,
            'swig_available': SWIG_AVAILABLE,
            'ipc_available': IPC_AVAILABLE,
            'recommendation': 'IPC' if IPC_AVAILABLE else 'Install kicad-python for IPC API'
        }

# Global bridge instance
api_bridge = APIBridge()

def get_api_bridge():
    """Get the global API bridge instance"""
    return api_bridge
