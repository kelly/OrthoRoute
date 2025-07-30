"""
OrthoRoute Plugin with IPC API Support
Updated to work with both SWIG and IPC APIs for future compatibility
"""

import wx
import os
import sys

# Add current directory to path for our modules
current_dir = os.path.dirname(__file__)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import API bridge
try:
    from api_bridge import get_api_bridge
    API_BRIDGE_AVAILABLE = True
except ImportError:
    API_BRIDGE_AVAILABLE = False
    print("⚠️ API Bridge not available, using direct SWIG API")

# Fallback to direct SWIG import
try:
    import pcbnew
    PCBNEW_AVAILABLE = True
except ImportError:
    PCBNEW_AVAILABLE = False
    print("❌ pcbnew not available")

# Import routing engine
try:
    from orthoroute_engine import OrthoRouteEngine
    ROUTING_ENGINE_AVAILABLE = True
except ImportError:
    ROUTING_ENGINE_AVAILABLE = False
    print("❌ OrthoRoute engine not available")

class OrthoRouteIPC(pcbnew.ActionPlugin):
    """OrthoRoute Plugin with IPC API Support"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter (IPC)"
        self.category = "Routing"
        self.description = "GPU-accelerated autorouter with IPC API support"
        self.show_toolbar_button = True
        
        # Set icon
        icon_path = os.path.join(os.path.dirname(__file__), 'icon.png')
        if os.path.exists(icon_path):
            self.icon_file_name = icon_path

    def Run(self):
        """Main plugin entry point"""
        print("\n" + "="*60)
        print("🚀 OrthoRoute GPU Autorouter with IPC API Support")
        print("="*60)
        
        # Check API availability
        api_info = self.check_api_availability()
        print(f"📊 API Status: {api_info}")
        
        if not PCBNEW_AVAILABLE:
            wx.MessageBox("KiCad pcbnew module not available!", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        if not ROUTING_ENGINE_AVAILABLE:
            wx.MessageBox("OrthoRoute engine not available!", "Error", wx.OK | wx.ICON_ERROR)
            return
        
        try:
            # Get board using API bridge or direct SWIG
            board = self.get_board()
            if not board:
                wx.MessageBox("No board loaded!", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            config = self.show_config_dialog()
            if not config:
                return  # User cancelled
            
            # Extract board data using appropriate API
            board_data = self.extract_board_data(board)
            if not board_data or not board_data.get('nets'):
                message = "No nets found to route!"
                print(f"❌ {message}")
                wx.MessageBox(message, "No Work", wx.OK | wx.ICON_WARNING)
                return
            
            print(f"✅ Found {len(board_data['nets'])} nets to route")
            
            # Run routing
            self.run_routing(board_data, config, board)
            
        except Exception as e:
            error_msg = f"OrthoRoute failed: {str(e)}"
            print(f"❌ {error_msg}")
            wx.MessageBox(error_msg, "Error", wx.OK | wx.ICON_ERROR)

    def check_api_availability(self):
        """Check which APIs are available"""
        if API_BRIDGE_AVAILABLE:
            bridge = get_api_bridge()
            return bridge.get_api_info()
        else:
            return {
                'current_api': 'SWIG',
                'swig_available': PCBNEW_AVAILABLE,
                'ipc_available': False,
                'recommendation': 'Install kicad-python for IPC API support'
            }

    def get_board(self):
        """Get board using available API"""
        if API_BRIDGE_AVAILABLE:
            bridge = get_api_bridge()
            return bridge.get_board()
        else:
            return pcbnew.GetBoard()

    def extract_board_data(self, board):
        """Extract board data using available API"""
        if API_BRIDGE_AVAILABLE:
            bridge = get_api_bridge()
            return bridge.extract_board_data(board)
        else:
            return self.extract_board_data_swig(board)

    def extract_board_data_swig(self, board):
        """Extract board data using SWIG API (fallback)"""
        try:
            # Board bounds
            bounds = board.GetBoardEdgesBoundingBox()
            board_data = {
                'bounds': {
                    'width_nm': bounds.GetWidth(),
                    'height_nm': bounds.GetHeight(),
                    'layers': board.GetCopperLayerCount()
                },
                'nets': [],
                'obstacles': {}
            }
            
            # Extract nets with corrected logic
            netcodes = board.GetNetsByNetcode()
            
            for netcode, net in netcodes.items():
                if netcode == 0:  # Skip unconnected
                    continue
                    
                net_name = net.GetNetname()
                if not net_name:
                    continue
                
                # Find pads for this net using corrected netcode comparison
                net_pads = []
                for footprint in board.GetFootprints():
                    for pad in footprint.Pads():
                        pad_net = pad.GetNet()
                        if pad_net.GetNetCode() == netcode:
                            pos = pad.GetPosition()
                            net_pads.append({
                                'x': pos.x,
                                'y': pos.y,
                                'layer': 0,
                                'pad_name': pad.GetName()
                            })
                
                if len(net_pads) >= 2:
                    board_data['nets'].append({
                        'id': netcode,
                        'name': net_name,
                        'pins': net_pads,
                        'width_nm': 200000,
                        'kicad_net': net
                    })
            
            return board_data
            
        except Exception as e:
            print(f"❌ SWIG board data extraction failed: {e}")
            return None

    def show_config_dialog(self):
        """Show routing configuration dialog"""
        dialog = OrthoRouteConfigDialog(None)
        
        if dialog.ShowModal() == wx.ID_OK:
            config = dialog.get_config()
            dialog.Destroy()
            return config
        else:
            dialog.Destroy()
            return None

    def run_routing(self, board_data, config, board):
        """Run the routing process"""
        try:
            # Create routing engine
            engine = OrthoRouteEngine()
            
            # Show progress dialog if requested
            progress_dialog = None
            if config.get('show_progress', True):
                progress_dialog = OrthoRouteProgressDialog(None)
                progress_dialog.Show()
            
            # Set up callbacks
            progress_callback = None
            cancel_callback = None
            
            if progress_dialog:
                cancel_callback = lambda: progress_dialog.should_cancel
                progress_callback = progress_dialog.update_progress
            
            routing_config = {
                **config,
                'progress_callback': progress_callback,
                'should_cancel': cancel_callback or (lambda: False)
            }
            
            # Run routing with board reference for track creation
            print("🔄 Starting routing process...")
            results = engine.route(board_data, routing_config, board=board)
            
            # Close progress dialog
            if progress_dialog:
                progress_dialog.Destroy()
            
            # Show results
            self.show_results(results)
            
        except Exception as e:
            if progress_dialog:
                progress_dialog.Destroy()
            raise e

    def show_results(self, results):
        """Show routing results"""
        if results['success']:
            stats = results['stats']
            message = f"""Routing completed!

Nets processed: {stats['total_nets']}
Successfully routed: {stats['successful_nets']}
Success rate: {stats['success_rate']:.1f}%
Tracks created: {len(results.get('tracks', []))}
Time: {stats['total_time_seconds']:.1f} seconds"""
            
            wx.MessageBox(message, "Routing Complete", wx.OK | wx.ICON_INFORMATION)
        else:
            error_msg = results.get('error', 'Unknown error')
            wx.MessageBox(f"Routing failed: {error_msg}", "Routing Failed", wx.OK | wx.ICON_ERROR)

class OrthoRouteConfigDialog(wx.Dialog):
    """Configuration dialog for OrthoRoute"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Configuration", 
                        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        
        self.create_controls()
        self.setup_layout()
        self.setup_defaults()

    def create_controls(self):
        """Create dialog controls"""
        # Grid pitch
        self.grid_pitch_label = wx.StaticText(self, label="Grid Pitch (mm):")
        self.grid_pitch_spin = wx.SpinCtrlDouble(self, value="0.1", min=0.05, max=1.0, inc=0.05)
        
        # Max iterations
        self.max_iter_label = wx.StaticText(self, label="Max Iterations:")
        self.max_iter_spin = wx.SpinCtrl(self, value="5", min=1, max=10)
        
        # Via cost
        self.via_cost_label = wx.StaticText(self, label="Via Cost:")
        self.via_cost_spin = wx.SpinCtrl(self, value="10", min=1, max=100)
        
        # Batch size
        self.batch_size_label = wx.StaticText(self, label="Batch Size:")
        self.batch_size_spin = wx.SpinCtrl(self, value="20", min=1, max=50)
        
        # Options
        self.show_progress_check = wx.CheckBox(self, label="Show Progress")
        self.debug_output_check = wx.CheckBox(self, label="Debug Output")
        
        # Buttons
        self.ok_button = wx.Button(self, wx.ID_OK, "Start Routing")
        self.cancel_button = wx.Button(self, wx.ID_CANCEL, "Cancel")

    def setup_layout(self):
        """Setup dialog layout"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Parameters
        param_sizer = wx.FlexGridSizer(4, 2, 5, 10)
        param_sizer.Add(self.grid_pitch_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_sizer.Add(self.grid_pitch_spin, 0, wx.EXPAND)
        param_sizer.Add(self.max_iter_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_sizer.Add(self.max_iter_spin, 0, wx.EXPAND)
        param_sizer.Add(self.via_cost_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_sizer.Add(self.via_cost_spin, 0, wx.EXPAND)
        param_sizer.Add(self.batch_size_label, 0, wx.ALIGN_CENTER_VERTICAL)
        param_sizer.Add(self.batch_size_spin, 0, wx.EXPAND)
        param_sizer.AddGrowableCol(1)
        
        # Options
        option_sizer = wx.BoxSizer(wx.VERTICAL)
        option_sizer.Add(self.show_progress_check, 0, wx.ALL, 5)
        option_sizer.Add(self.debug_output_check, 0, wx.ALL, 5)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self.ok_button, 0, wx.ALL, 5)
        button_sizer.Add(self.cancel_button, 0, wx.ALL, 5)
        
        # Main layout
        main_sizer.Add(param_sizer, 0, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(option_sizer, 0, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(button_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        self.SetSizer(main_sizer)
        self.Fit()

    def setup_defaults(self):
        """Setup default values"""
        self.show_progress_check.SetValue(True)
        self.debug_output_check.SetValue(False)

    def get_config(self):
        """Get configuration from dialog"""
        return {
            'grid_pitch_mm': self.grid_pitch_spin.GetValue(),
            'max_iterations': self.max_iter_spin.GetValue(),
            'via_cost': self.via_cost_spin.GetValue(),
            'batch_size': self.batch_size_spin.GetValue(),
            'show_progress': self.show_progress_check.GetValue(),
            'debug_output': self.debug_output_check.GetValue()
        }

class OrthoRouteProgressDialog(wx.Dialog):
    """Progress dialog for routing operations"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Progress", 
                        style=wx.DEFAULT_DIALOG_STYLE | wx.STAY_ON_TOP)
        
        self.should_cancel = False
        self.create_controls()
        self.setup_layout()

    def create_controls(self):
        """Create progress controls"""
        self.status_text = wx.StaticText(self, label="Initializing...")
        self.progress_gauge = wx.Gauge(self, range=100)
        self.cancel_button = wx.Button(self, wx.ID_CANCEL, "Cancel")
        
        self.cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)

    def setup_layout(self):
        """Setup progress dialog layout"""
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(self.status_text, 0, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(self.progress_gauge, 0, wx.ALL | wx.EXPAND, 10)
        main_sizer.Add(self.cancel_button, 0, wx.ALL | wx.CENTER, 10)
        
        self.SetSizer(main_sizer)
        self.SetSize((300, 120))

    def update_progress(self, progress_info):
        """Update progress display"""
        if 'current_net' in progress_info:
            self.status_text.SetLabel(f"Routing: {progress_info['current_net']}")
        
        if 'progress' in progress_info:
            self.progress_gauge.SetValue(int(progress_info['progress']))
        
        self.Update()
        wx.SafeYield()

    def on_cancel(self, event):
        """Handle cancel button"""
        self.should_cancel = True
        self.status_text.SetLabel("Cancelling...")
        self.cancel_button.Enable(False)

# Register the plugin
if PCBNEW_AVAILABLE:
    OrthoRouteIPC().register()
else:
    print("❌ Cannot register OrthoRoute plugin - pcbnew not available")
