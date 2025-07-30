"""
OrthoRoute GPU Autorouter - Enhanced Version
Compact UI with GPU detection and system information.
"""

import pcbnew
import wx
import traceback
import os
from .orthoroute_engine import OrthoRouteEngine

def get_gpu_info():
    """Get GPU information for display"""
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        mem_free, mem_total = device.mem_info
        
        return {
            'available': True,
            'name': props['name'].decode('utf-8'),
            'memory_total_gb': mem_total / 1024**3,
            'memory_free_gb': mem_free / 1024**3,
            'compute_capability': f"{props['major']}.{props['minor']}"
        }
    except Exception:
        return {
            'available': False,
            'name': 'No CUDA GPU detected',
            'memory_total_gb': 0,
            'memory_free_gb': 0,
            'compute_capability': 'N/A'
        }

class OrthoRouteConfigDialog(wx.Dialog):
    """Compact configuration dialog for OrthoRoute"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute GPU Autorouter", size=(480, 420))
        
        panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title with icon
        title_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Load icon if available
        icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
        if os.path.exists(icon_path):
            try:
                icon_bitmap = wx.Bitmap(icon_path, wx.BITMAP_TYPE_PNG)
                scaled_bitmap = wx.Bitmap(icon_bitmap.ConvertToImage().Scale(24, 24, wx.IMAGE_QUALITY_HIGH))
                icon_ctrl = wx.StaticBitmap(panel, bitmap=scaled_bitmap)
                title_sizer.Add(icon_ctrl, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            except:
                pass
        
        title = wx.StaticText(panel, label="OrthoRoute GPU Autorouter")
        title_font = title.GetFont()
        title_font.SetPointSize(12)
        title_font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(title_font)
        title_sizer.Add(title, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        main_sizer.Add(title_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        # GPU/CPU Selection and Info
        gpu_info = get_gpu_info()
        
        hardware_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Hardware")
        
        # GPU/CPU choice
        hardware_choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        hardware_choice_sizer.Add(wx.StaticText(panel, label="Processing:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        
        if gpu_info['available']:
            choices = ["GPU (Recommended)", "CPU (Fallback)"]
            self.hardware_choice = wx.Choice(panel, choices=choices)
            self.hardware_choice.SetSelection(0)
        else:
            choices = ["CPU Only"]
            self.hardware_choice = wx.Choice(panel, choices=choices)
            self.hardware_choice.SetSelection(0)
            
        hardware_choice_sizer.Add(self.hardware_choice, 1, wx.ALL, 5)
        hardware_box.Add(hardware_choice_sizer, 0, wx.ALL | wx.EXPAND, 5)
        
        # GPU info display
        if gpu_info['available']:
            gpu_info_text = f"GPU: {gpu_info['name'][:30]}...\nMemory: {gpu_info['memory_free_gb']:.1f}GB free / {gpu_info['memory_total_gb']:.1f}GB total"
            self.gpu_info_label = wx.StaticText(panel, label=gpu_info_text)
            font = self.gpu_info_label.GetFont()
            font.SetPointSize(8)
            self.gpu_info_label.SetFont(font)
            hardware_box.Add(self.gpu_info_label, 0, wx.ALL, 5)
        else:
            cpu_info_text = "CPU mode only (CuPy/CUDA not available)\nRouting will be slower but functional"
            self.gpu_info_label = wx.StaticText(panel, label=cpu_info_text)
            font = self.gpu_info_label.GetFont()
            font.SetPointSize(8)
            self.gpu_info_label.SetFont(font)
            hardware_box.Add(self.gpu_info_label, 0, wx.ALL, 5)
        
        main_sizer.Add(hardware_box, 0, wx.ALL | wx.EXPAND, 10)
        
        # Routing Parameters (compact)
        params_box = wx.StaticBoxSizer(wx.VERTICAL, panel, "Routing Parameters")
        
        # First row: Grid pitch and iterations
        row1_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        row1_sizer.Add(wx.StaticText(panel, label="Grid (mm):"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.grid_pitch_ctrl = wx.SpinCtrlDouble(panel, size=(80, -1))
        self.grid_pitch_ctrl.SetValue(0.1)
        self.grid_pitch_ctrl.SetRange(0.05, 1.0)
        self.grid_pitch_ctrl.SetIncrement(0.05)
        row1_sizer.Add(self.grid_pitch_ctrl, 0, wx.ALL, 5)
        
        row1_sizer.Add(wx.StaticText(panel, label="Iterations:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.max_iter_ctrl = wx.SpinCtrl(panel, size=(60, -1))
        self.max_iter_ctrl.SetValue(3)
        self.max_iter_ctrl.SetRange(1, 10)
        row1_sizer.Add(self.max_iter_ctrl, 0, wx.ALL, 5)
        
        params_box.Add(row1_sizer, 0, wx.ALL, 5)
        
        # Second row: Via cost and batch size
        row2_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        row2_sizer.Add(wx.StaticText(panel, label="Via Cost:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.via_cost_ctrl = wx.SpinCtrl(panel, size=(60, -1))
        self.via_cost_ctrl.SetValue(10)
        self.via_cost_ctrl.SetRange(1, 100)
        row2_sizer.Add(self.via_cost_ctrl, 0, wx.ALL, 5)
        
        row2_sizer.Add(wx.StaticText(panel, label="Batch:"), 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
        self.batch_size_ctrl = wx.SpinCtrl(panel, size=(60, -1))
        self.batch_size_ctrl.SetValue(20 if gpu_info['available'] else 5)
        self.batch_size_ctrl.SetRange(1, 50)
        row2_sizer.Add(self.batch_size_ctrl, 0, wx.ALL, 5)
        
        params_box.Add(row2_sizer, 0, wx.ALL, 5)
        
        main_sizer.Add(params_box, 0, wx.ALL | wx.EXPAND, 10)
        
        # Options (compact)
        options_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.progress_cb = wx.CheckBox(panel, label="Show progress")
        self.progress_cb.SetValue(True)
        options_sizer.Add(self.progress_cb, 0, wx.ALL, 5)
        
        self.debug_cb = wx.CheckBox(panel, label="Debug mode")
        self.debug_cb.SetValue(False)
        options_sizer.Add(self.debug_cb, 0, wx.ALL, 5)
        
        main_sizer.Add(options_sizer, 0, wx.ALL | wx.CENTER, 5)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.start_btn = wx.Button(panel, wx.ID_OK, "🚀 Start Routing")
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel")
        
        button_sizer.Add(self.start_btn, 0, wx.ALL, 5)
        button_sizer.Add(self.cancel_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(main_sizer)
        self.Centre()


class OrthoRouteProgressDialog(wx.Dialog):
    """Modern progress dialog with detailed feedback"""
    
    def __init__(self, parent):
        super().__init__(parent, title="OrthoRoute Progress", size=(450, 250))
        
        panel = wx.Panel(self)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Status text
        self.status_text = wx.StaticText(panel, label="Initializing router...")
        font = self.status_text.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.status_text.SetFont(font)
        sizer.Add(self.status_text, 0, wx.ALL | wx.EXPAND, 10)
        
        # Progress bar
        self.progress_gauge = wx.Gauge(panel, range=100)
        sizer.Add(self.progress_gauge, 0, wx.ALL | wx.EXPAND, 10)
        
        # Current net info
        self.net_text = wx.StaticText(panel, label="")
        sizer.Add(self.net_text, 0, wx.ALL | wx.EXPAND, 10)
        
        # Statistics
        self.stats_text = wx.StaticText(panel, label="Nets: 0/0 | Success: 0% | Tracks: 0")
        font = self.stats_text.GetFont()
        font.SetPointSize(9)
        self.stats_text.SetFont(font)
        sizer.Add(self.stats_text, 0, wx.ALL | wx.EXPAND, 5)
        
        # Cancel button
        self.cancel_btn = wx.Button(panel, wx.ID_CANCEL, "Cancel Routing")
        sizer.Add(self.cancel_btn, 0, wx.ALL | wx.CENTER, 10)
        
        panel.SetSizer(sizer)
        self.Centre()
        
        self.should_cancel = False
        self.Bind(wx.EVT_BUTTON, self.on_cancel, self.cancel_btn)
    
    def on_cancel(self, event):
        self.should_cancel = True
        self.status_text.SetLabel("⚠️ Cancelling routing...")
        self.cancel_btn.Enable(False)
        
    def update_progress(self, data):
        if not self.should_cancel:
            progress = data.get('progress', 0)
            current_net = data.get('current_net', '')
            stage = data.get('stage', '')
            nets_processed = data.get('nets_processed', 0)
            total_nets = data.get('total_nets', 0)
            success_rate = data.get('success_rate', 0)
            tracks_created = data.get('tracks_created', 0)
            
            self.progress_gauge.SetValue(int(progress))
            
            if current_net:
                self.net_text.SetLabel(f"Current: {current_net}")
            
            if stage:
                self.status_text.SetLabel(f"🔄 {stage}")
            
            self.stats_text.SetLabel(
                f"Nets: {nets_processed}/{total_nets} | Success: {success_rate:.1f}% | Tracks: {tracks_created}"
            )
            
            wx.GetApp().Yield()


class OrthoRoutePlugin(pcbnew.ActionPlugin):
    """OrthoRoute GPU Autorouter Plugin - Enhanced Edition"""
    
    def defaults(self):
        self.name = "OrthoRoute GPU Autorouter"
        self.category = "Routing"
        self.description = "GPU-accelerated PCB autorouter with Lee's algorithm"
        self.show_toolbar_button = True
        
        # Fix icon path - use absolute path from plugin directory
        plugin_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(plugin_dir, "icon.png")
        
        if os.path.exists(icon_path):
            self.icon_file_name = icon_path
            print(f"✅ OrthoRoute icon loaded: {icon_path}")
        else:
            print(f"⚠️ OrthoRoute icon not found: {icon_path}")
        
    def Run(self):
        """Main plugin entry point with enhanced error handling"""
        try:
            print("🚀 OrthoRoute GPU Autorouter starting...")
            
            # Get the current board
            board = pcbnew.GetBoard()
            if not board:
                wx.MessageBox("No board loaded!", "Error", wx.OK | wx.ICON_ERROR)
                return
            
            # Show configuration dialog
            config_dialog = OrthoRouteConfigDialog(None)
            if config_dialog.ShowModal() != wx.ID_OK:
                config_dialog.Destroy()
                return
            
            # Get configuration
            config = {
                'grid_pitch_mm': config_dialog.grid_pitch_ctrl.GetValue(),
                'max_iterations': config_dialog.max_iter_ctrl.GetValue(),
                'via_cost': config_dialog.via_cost_ctrl.GetValue(),
                'batch_size': config_dialog.batch_size_ctrl.GetValue(),
                'use_gpu': config_dialog.hardware_choice.GetSelection() == 0 and get_gpu_info()['available'],
                'show_progress': config_dialog.progress_cb.GetValue(),
                'debug_output': config_dialog.debug_cb.GetValue()
            }
            
            config_dialog.Destroy()
            
            print(f"🔧 Configuration: Grid={config['grid_pitch_mm']}mm, Iterations={config['max_iterations']}, GPU={config['use_gpu']}")
            
            # Create progress dialog if requested
            progress_dialog = None
            if config['show_progress']:
                progress_dialog = OrthoRouteProgressDialog(None)
                progress_dialog.Show()
                progress_dialog.update_progress({
                    'stage': 'Analyzing board...',
                    'progress': 5
                })
            
            # Extract board data with enhanced debugging
            print("📊 Extracting board data...")
            board_data = self.extract_board_data(board)
            
            # Enhanced board validation
            if not board_data['nets']:
                message = f"No nets found to route!\n\nBoard Analysis:\n• Footprints: {board_data.get('footprint_count', 0)}\n• Total nets: {board_data.get('total_nets', 0)}\n• Routeable nets: {len(board_data['nets'])}"
                print(f"❌ {message}")
                if progress_dialog:
                    progress_dialog.Destroy()
                wx.MessageBox(message, "No Work Found", wx.OK | wx.ICON_WARNING)
                return
            
            print(f"✅ Found {len(board_data['nets'])} nets to route")
            
            if progress_dialog:
                progress_dialog.update_progress({
                    'stage': 'Initializing routing engine...',
                    'progress': 10,
                    'total_nets': len(board_data['nets'])
                })
            
            # Create routing engine
            engine = OrthoRouteEngine()
            if config['debug_output']:
                engine.debug_print = print
            
            # Set up progress callback
            cancel_callback = None
            progress_callback = None
            
            if progress_dialog:
                cancel_callback = lambda: progress_dialog.should_cancel
                progress_callback = progress_dialog.update_progress
            
            routing_config = {
                **config,
                'progress_callback': progress_callback,
                'should_cancel': cancel_callback or (lambda: False)
            }
            
            # Start routing WITH BOARD REFERENCE
            print("🔄 Starting routing process...")
            if progress_dialog:
                progress_dialog.update_progress({
                    'stage': 'Routing nets...',
                    'progress': 15
                })
            
            results = engine.route(board_data, routing_config, board=board)
            
            # Close progress dialog
            if progress_dialog:
                progress_dialog.Destroy()
            
            # Show enhanced results
            if results['success']:
                stats = results['stats']
                tracks_created = len(results.get('tracks', []))
                
                # Calculate actual success metrics
                total_nets = stats.get('total_nets', 0)
                successful_nets = stats.get('successful_nets', 0)
                success_rate = (successful_nets / total_nets * 100) if total_nets > 0 else 0
                
                message = (f"🎉 Routing completed!\n\n"
                          f"📊 Results:\n"
                          f"• Nets processed: {total_nets}\n"
                          f"• Successfully routed: {successful_nets}\n" 
                          f"• Success rate: {success_rate:.1f}%\n"
                          f"• Tracks created: {tracks_created}\n"
                          f"• Processing time: {stats.get('total_time_seconds', 0):.2f} seconds\n"
                          f"• Hardware: {'GPU' if config['use_gpu'] else 'CPU'}")
                
                print(f"✅ {message}")
                wx.MessageBox(message, "Routing Complete", wx.OK | wx.ICON_INFORMATION)
                
                # Force board refresh
                try:
                    pcbnew.Refresh()
                    board.BuildListOfNets()  # Rebuild net list
                    print("🔄 Board display refreshed")
                except Exception as e:
                    print(f"⚠️ Display refresh warning: {e}")
                
            else:
                error_msg = results.get('error', 'Unknown error occurred')
                print(f"❌ Routing failed: {error_msg}")
                
                # Show detailed error message
                detailed_msg = f"Routing failed: {error_msg}\n\nDebugging info:\n• Board loaded: Yes\n• Nets found: {len(board_data['nets'])}\n• GPU available: {get_gpu_info()['available']}"
                wx.MessageBox(detailed_msg, "Routing Failed", wx.OK | wx.ICON_ERROR)
                
        except Exception as e:
            error_msg = f"Plugin error: {str(e)}\n\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            wx.MessageBox(error_msg, "Plugin Error", wx.OK | wx.ICON_ERROR)
    
    def extract_board_data(self, board):
        """Extract board data for routing with enhanced debugging"""
        
        # Get board bounds
        bounds = board.GetBoardEdgesBoundingBox()
        width_nm = bounds.GetWidth()
        height_nm = bounds.GetHeight()
        
        min_x_nm = bounds.GetX()
        min_y_nm = bounds.GetY()
        max_x_nm = min_x_nm + width_nm
        max_y_nm = min_y_nm + height_nm
        
        # Get layer count
        layer_count = board.GetCopperLayerCount()
        
        print(f"📐 Board: {width_nm/1e6:.1f}mm x {height_nm/1e6:.1f}mm, {layer_count} layers")
        
        # Count footprints and pads
        footprints = list(board.GetFootprints())
        total_pads = sum(len(list(fp.Pads())) for fp in footprints)
        
        print(f"📊 Board components: {len(footprints)} footprints, {total_pads} total pads")
        
        # Extract nets with ENHANCED net-pad matching
        nets_data = []
        netcodes = board.GetNetsByNetcode()
        
        print(f"🔍 Found {len(netcodes)} total nets in board")
        
        # Debug first few nets
        for i, (netcode, kicad_net) in enumerate(list(netcodes.items())[:5]):
            net_name = kicad_net.GetNetname()
            print(f"   Debug net {i+1}: {netcode} = '{net_name}'")
        
        routeable_nets = 0
        total_pins_found = 0
        
        for netcode, kicad_net in netcodes.items():
            net_name = kicad_net.GetNetname()
            
            if netcode == 0:  # Skip unconnected net
                continue
                
            if not net_name or net_name.startswith('$'):  # Skip system nets
                continue
            
            # Get all pads for this net - ENHANCED VERSION
            pins = []
            
            # CORRECT approach: iterate through all footprints and pads
            for footprint in footprints:
                footprint_ref = footprint.GetReference()
                for pad in footprint.Pads():
                    # FIXED: Compare netcodes instead of net objects
                    pad_net = pad.GetNet()
                    if pad_net and pad_net.GetNetCode() == netcode:
                        pos = pad.GetPosition()
                        layer = pad.GetLayer()
                        pad_name = pad.GetName()
                        
                        # Convert layer to internal layer number (robust)
                        internal_layer = 0  # Default to front copper
                        try:
                            if layer == pcbnew.F_Cu:
                                internal_layer = 0
                            elif layer == pcbnew.B_Cu:
                                internal_layer = 1 if layer_count >= 2 else 0
                            elif hasattr(pcbnew, 'In1_Cu') and layer >= pcbnew.In1_Cu:
                                internal_layer = min(layer - pcbnew.In1_Cu + 2, layer_count - 1)
                        except:
                            internal_layer = 0  # Fallback to front copper
                        
                        pins.append({
                            'x': int(pos.x),
                            'y': int(pos.y), 
                            'layer': internal_layer
                        })
                        total_pins_found += 1
            
            # Include nets with 2+ pins
            if len(pins) >= 2:
                nets_data.append({
                    'id': netcode,
                    'name': net_name,
                    'pins': pins,
                    'kicad_net': kicad_net,  # Store KiCad net reference
                    'width_nm': 200000  # Default 0.2mm track width
                })
                routeable_nets += 1
                
                if routeable_nets <= 3:  # Debug first few routeable nets
                    print(f"     ✅ Net '{net_name}': {len(pins)} pins")
                    for j, pin in enumerate(pins[:2]):
                        print(f"       Pin {j+1}: ({pin['x']/1e6:.2f}, {pin['y']/1e6:.2f})mm, layer {pin['layer']}")
        
        print(f"� FINAL ANALYSIS:")
        print(f"   • Total footprints: {len(footprints)}")
        print(f"   • Total pads: {total_pads}")
        print(f"   • Total nets: {len(netcodes)}")
        print(f"   • Pins assigned to nets: {total_pins_found}")
        print(f"   • Routeable nets: {routeable_nets}")
        
        # Enhanced debugging for zero nets case
        if routeable_nets == 0:
            print("🚨 ZERO NETS DEBUG:")
            print("   Checking if pads have nets assigned...")
            
            unassigned_pads = 0
            for fp in footprints[:3]:  # Check first few footprints
                print(f"   Footprint: {fp.GetReference()}")
                for pad in list(fp.Pads())[:3]:  # Check first few pads
                    pad_net = pad.GetNet()
                    if pad_net:
                        netcode = pad_net.GetNetCode()
                        net_name = pad_net.GetNetname()
                        print(f"     Pad {pad.GetName()}: net {netcode} '{net_name}'")
                    else:
                        unassigned_pads += 1
                        print(f"     Pad {pad.GetName()}: NO NET ASSIGNED")
            
            print(f"   Unassigned pads found: {unassigned_pads}")
            
            if unassigned_pads > 0:
                print("   💡 SUGGESTION: Run 'Update PCB from Schematic' in KiCad")
        
        board_data = {
            'bounds': {
                'width_nm': width_nm,
                'height_nm': height_nm,
                'min_x_nm': min_x_nm,
                'min_y_nm': min_y_nm,
                'max_x_nm': max_x_nm,
                'max_y_nm': max_y_nm,
                'layers': layer_count
            },
            'nets': nets_data,
            'obstacles': {},  # TODO: Extract existing tracks/vias
            'footprint_count': len(footprints),
            'total_nets': len(netcodes),
            'total_pads': total_pads
        }
        
        return board_data


# Register the plugin
OrthoRoutePlugin().register()
