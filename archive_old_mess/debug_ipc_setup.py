#!/usr/bin/env python3
"""
KiCad IPC API Debug Setup and Testing Tool
Sets up proper debugging environment and tests IPC connection
"""

import os
import sys
import json
import tempfile
from pathlib import Path

def setup_debug_environment():
    """Set up KiCad debugging environment variables"""
    print("🔧 Setting up KiCad debugging environment...")
    
    debug_vars = {
        'KICAD_ALLOC_CONSOLE': '1',      # Windows - show console output
        'KICAD_ENABLE_WXTRACE': '1',     # Enable tracing in release builds
        'WXTRACE': 'KICAD_API'           # Enable API subsystem tracing
    }
    
    for var, value in debug_vars.items():
        os.environ[var] = value
        print(f"  {var}={value}")
    
    print("✅ Debug environment variables set")
    return debug_vars

def enable_api_logging():
    """Enable KiCad API logging by creating/updating kicad_advanced config"""
    print("📝 Enabling KiCad API logging...")
    
    # Find KiCad config directory
    if sys.platform == "win32":
        config_base = Path.home() / "Documents" / "KiCad"
    elif sys.platform == "darwin":
        config_base = Path.home() / "Documents" / "KiCad"
    else:  # Linux
        config_base = Path.home() / ".local" / "share" / "KiCad"
    
    # Look for version directories
    version_dirs = [d for d in config_base.glob("*") if d.is_dir() and d.name.replace(".", "").isdigit()]
    
    if not version_dirs:
        print(f"❌ No KiCad version directories found in {config_base}")
        return False
    
    # Use the highest version directory
    latest_version = max(version_dirs, key=lambda x: tuple(map(int, x.name.split("."))))
    config_dir = latest_version
    
    print(f"📁 Using KiCad config directory: {config_dir}")
    
    # Create/update kicad_advanced file
    advanced_config = config_dir / "kicad_advanced"
    
    try:
        # Read existing config if it exists
        existing_lines = []
        if advanced_config.exists():
            with open(advanced_config, 'r') as f:
                existing_lines = f.readlines()
        
        # Check if EnableAPILogging is already set
        api_logging_set = any('EnableAPILogging' in line for line in existing_lines)
        
        if not api_logging_set:
            # Add EnableAPILogging=1
            with open(advanced_config, 'a') as f:
                if existing_lines and not existing_lines[-1].endswith('\n'):
                    f.write('\n')
                f.write('EnableAPILogging=1\n')
            print(f"✅ Added EnableAPILogging=1 to {advanced_config}")
        else:
            print(f"✅ EnableAPILogging already configured in {advanced_config}")
        
        # Show where log file will be created
        log_file = config_dir / "logs" / "api.log"
        print(f"📋 API log file will be created at: {log_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to configure API logging: {e}")
        return False

def test_ipc_connection():
    """Test IPC connection to KiCad"""
    print("🔌 Testing KiCad IPC API connection...")
    
    # Check for environment variables
    api_socket = os.environ.get('KICAD_API_SOCKET')
    api_token = os.environ.get('KICAD_API_TOKEN')
    
    print(f"  KICAD_API_SOCKET: {api_socket or 'Not set'}")
    print(f"  KICAD_API_TOKEN: {'Set' if api_token else 'Not set'}")
    
    if not api_socket or not api_token:
        print("ℹ️  Environment variables not set - this is normal when not launched by KiCad")
        print("   To test with KiCad, create an IPC plugin and launch from KiCad")
        return False
    
    try:
        from kipy import KiCad
        kicad = KiCad(socket_path=api_socket, token=api_token)
        print("✅ Successfully connected to KiCad via IPC API")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to KiCad: {e}")
        return False

def create_plugin_directory_structure():
    """Create proper IPC plugin directory structure"""
    print("📁 Creating IPC plugin directory structure...")
    
    # Find KiCad documents directory
    if sys.platform == "win32":
        kicad_docs = Path.home() / "Documents" / "KiCad"
    elif sys.platform == "darwin":
        kicad_docs = Path.home() / "Documents" / "KiCad"
    else:  # Linux
        kicad_docs = Path.home() / ".local" / "share" / "KiCad"
    
    # Look for version directories
    version_dirs = [d for d in kicad_docs.glob("*") if d.is_dir() and d.name.replace(".", "").isdigit()]
    
    if not version_dirs:
        print(f"❌ No KiCad version directories found in {kicad_docs}")
        return None
    
    # Use the highest version directory
    latest_version = max(version_dirs, key=lambda x: tuple(map(int, x.name.split("."))))
    plugins_dir = latest_version / "plugins" / "orthoroute"
    
    print(f"📁 Plugin directory: {plugins_dir}")
    
    # Create directory structure
    try:
        plugins_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created plugin directory: {plugins_dir}")
        return plugins_dir
    except Exception as e:
        print(f"❌ Failed to create plugin directory: {e}")
        return None

def show_debug_instructions():
    """Show instructions for debugging KiCad IPC plugins"""
    print("\n" + "="*60)
    print("🐛 KICAD IPC PLUGIN DEBUGGING INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Enable Debug Environment (run this script first):")
    print("   python debug_ipc_setup.py")
    
    print("\n2. Launch KiCad with Debug Output:")
    print("   - Windows: Launch KiCad normally (console will appear)")
    print("   - Linux/macOS: Launch KiCad from terminal to see output")
    
    print("\n3. Check API Log File:")
    if sys.platform == "win32":
        log_path = Path.home() / "Documents" / "KiCad" / "9.0" / "logs" / "api.log"
    else:
        log_path = "~/.local/share/KiCad/9.0/logs/api.log"
    print(f"   {log_path}")
    
    print("\n4. Install Plugin:")
    print("   - Copy plugin files to proper IPC plugin directory")
    print("   - Restart KiCad to detect new plugin")
    
    print("\n5. Test Plugin:")
    print("   - Look for 'OrthoRoute GPU Autorouter' in Tools menu")
    print("   - Check console output and API log for detailed info")
    
    print("\n6. Environment Variables (set by KiCad when launching IPC plugins):")
    print("   KICAD_API_SOCKET - Path to Unix socket/named pipe")
    print("   KICAD_API_TOKEN  - Authentication token")

def main():
    """Main debug setup function"""
    print("🚀 KiCad IPC API Debug Setup Tool")
    print("="*50)
    
    # Setup debug environment
    setup_debug_environment()
    
    # Enable API logging
    enable_api_logging()
    
    # Test IPC connection (will fail if not launched by KiCad)
    test_ipc_connection()
    
    # Create plugin directory structure
    plugin_dir = create_plugin_directory_structure()
    
    # Show debugging instructions
    show_debug_instructions()
    
    print("\n✅ Debug setup complete!")
    print("   Launch KiCad to test with debug output enabled")

if __name__ == "__main__":
    main()
