#!/usr/bin/env python3
"""
OrthoRoute Configuration Dialog
GUI for setting routing parameters
"""

import logging

logger = logging.getLogger(__name__)

# Try to import wxPython for GUI
try:
    import wx
    WX_AVAILABLE = True
except ImportError:
    WX_AVAILABLE = False
    logger.warning("wxPython not available, using CLI configuration")

class OrthoRouteConfigDialog:
    """Configuration dialog for OrthoRoute parameters"""
    
    def __init__(self, parent=None):
        self.config = self._get_default_config()
        if WX_AVAILABLE:
            self.dialog = self._create_wx_dialog(parent)
        else:
            self.dialog = None
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            'grid_pitch': 0.1,      # mm
            'max_iterations': 5,     # routing attempts
            'via_cost': 50,         # penalty for vias
            'use_gpu': True,        # enable GPU acceleration
            'show_progress': True,  # show progress dialog
            'debug_mode': False     # enable debug output
        }
    
    def show_modal(self):
        """Show configuration dialog and return config if accepted"""
        if WX_AVAILABLE and self.dialog:
            return self._show_wx_dialog()
        else:
            return self._show_cli_dialog()
    
    def _create_wx_dialog(self, parent):
        """Create wxPython dialog"""
        if not WX_AVAILABLE:
            return None
        
        dialog = wx.Dialog(parent, wx.ID_ANY, "OrthoRoute Configuration", 
                          size=(400, 300))
        
        # Create main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Grid pitch control
        grid_box = wx.StaticBox(dialog, wx.ID_ANY, "Grid Settings")
        grid_sizer = wx.StaticBoxSizer(grid_box, wx.VERTICAL)
        
        grid_pitch_sizer = wx.BoxSizer(wx.HORIZONTAL)
        grid_pitch_sizer.Add(wx.StaticText(dialog, wx.ID_ANY, "Grid Pitch (mm):"), 
                            0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.grid_pitch_ctrl = wx.SpinCtrlDouble(dialog, wx.ID_ANY, 
                                               value=str(self.config['grid_pitch']),
                                               min=0.01, max=1.0, inc=0.01)
        grid_pitch_sizer.Add(self.grid_pitch_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        grid_sizer.Add(grid_pitch_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        # Max iterations control
        iter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        iter_sizer.Add(wx.StaticText(dialog, wx.ID_ANY, "Max Iterations:"), 
                      0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.max_iter_ctrl = wx.SpinCtrl(dialog, wx.ID_ANY, 
                                        value=str(self.config['max_iterations']),
                                        min=1, max=20)
        iter_sizer.Add(self.max_iter_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        grid_sizer.Add(iter_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(grid_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Algorithm settings
        algo_box = wx.StaticBox(dialog, wx.ID_ANY, "Algorithm Settings")
        algo_sizer = wx.StaticBoxSizer(algo_box, wx.VERTICAL)
        
        via_cost_sizer = wx.BoxSizer(wx.HORIZONTAL)
        via_cost_sizer.Add(wx.StaticText(dialog, wx.ID_ANY, "Via Cost:"), 
                          0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        self.via_cost_ctrl = wx.SpinCtrl(dialog, wx.ID_ANY, 
                                        value=str(self.config['via_cost']),
                                        min=1, max=100)
        via_cost_sizer.Add(self.via_cost_ctrl, 1, wx.EXPAND | wx.ALL, 5)
        algo_sizer.Add(via_cost_sizer, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(algo_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Options
        options_box = wx.StaticBox(dialog, wx.ID_ANY, "Options")
        options_sizer = wx.StaticBoxSizer(options_box, wx.VERTICAL)
        
        self.gpu_cb = wx.CheckBox(dialog, wx.ID_ANY, "Use GPU Acceleration")
        self.gpu_cb.SetValue(self.config['use_gpu'])
        options_sizer.Add(self.gpu_cb, 0, wx.EXPAND | wx.ALL, 5)
        
        self.progress_cb = wx.CheckBox(dialog, wx.ID_ANY, "Show Progress Dialog")
        self.progress_cb.SetValue(self.config['show_progress'])
        options_sizer.Add(self.progress_cb, 0, wx.EXPAND | wx.ALL, 5)
        
        self.debug_cb = wx.CheckBox(dialog, wx.ID_ANY, "Debug Mode")
        self.debug_cb.SetValue(self.config['debug_mode'])
        options_sizer.Add(self.debug_cb, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(options_sizer, 0, wx.EXPAND | wx.ALL, 10)
        
        # Buttons
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        ok_btn = wx.Button(dialog, wx.ID_OK, "Route Board")
        cancel_btn = wx.Button(dialog, wx.ID_CANCEL, "Cancel")
        button_sizer.Add(ok_btn, 0, wx.ALL, 5)
        button_sizer.Add(cancel_btn, 0, wx.ALL, 5)
        
        main_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER | wx.ALL, 10)
        
        dialog.SetSizer(main_sizer)
        dialog.Layout()
        
        return dialog
    
    def _show_wx_dialog(self):
        """Show wxPython dialog and return config"""
        if self.dialog.ShowModal() == wx.ID_OK:
            config = {
                'grid_pitch': self.grid_pitch_ctrl.GetValue(),
                'max_iterations': self.max_iter_ctrl.GetValue(),
                'via_cost': self.via_cost_ctrl.GetValue(),
                'use_gpu': self.gpu_cb.GetValue(),
                'show_progress': self.progress_cb.GetValue(),
                'debug_mode': self.debug_cb.GetValue()
            }
            self.dialog.Destroy()
            return config
        else:
            self.dialog.Destroy()
            return None
    
    def _show_cli_dialog(self):
        """Show command-line configuration"""
        print("\n" + "="*50)
        print("OrthoRoute Configuration")
        print("="*50)
        
        try:
            # Grid pitch
            grid_pitch = input(f"Grid pitch in mm [{self.config['grid_pitch']}]: ").strip()
            if grid_pitch:
                self.config['grid_pitch'] = float(grid_pitch)
            
            # Max iterations
            max_iter = input(f"Max iterations [{self.config['max_iterations']}]: ").strip()
            if max_iter:
                self.config['max_iterations'] = int(max_iter)
            
            # Via cost
            via_cost = input(f"Via cost [{self.config['via_cost']}]: ").strip()
            if via_cost:
                self.config['via_cost'] = int(via_cost)
            
            # GPU usage
            use_gpu = input(f"Use GPU acceleration? [{'Y' if self.config['use_gpu'] else 'N'}]: ").strip().lower()
            if use_gpu in ['y', 'yes', 'n', 'no']:
                self.config['use_gpu'] = use_gpu in ['y', 'yes']
            
            # Confirm
            confirm = input("\nStart routing with these settings? [Y/n]: ").strip().lower()
            if confirm in ['', 'y', 'yes']:
                return self.config
            else:
                return None
                
        except (ValueError, KeyboardInterrupt):
            print("\nConfiguration cancelled.")
            return None
