"""
Installation script for OrthoRoute Native IPC Plugin
Automatically installs the plugin to the correct KiCad directory
"""

import os
import sys
import shutil
import platform
from pathlib import Path

def find_kicad_version():
    """Try to detect KiCad version from common installation paths"""
    system = platform.system()
    
    if system == "Windows":
        # Check common KiCad installation paths
        program_files = os.environ.get('PROGRAMFILES', 'C:\\Program Files')
        kicad_paths = [
            Path(program_files) / "KiCad" / "9.0",
            Path(program_files) / "KiCad" / "8.0",
            Path(program_files) / "KiCad" / "7.0"
        ]
        
        for path in kicad_paths:
            if path.exists():
                return path.name
                
    # Default to 9.0 if not found
    return "9.0"

def get_kicad_plugins_dir():
    """Get the KiCad plugins directory for the current user"""
    system = platform.system()
    version = find_kicad_version()
    
    if system == "Windows":
        docs_path = Path.home() / "Documents" / "KiCad" / version / "plugins"
    elif system == "Darwin":  # macOS
        docs_path = Path.home() / "Documents" / "KiCad" / version / "plugins"
    else:  # Linux
        docs_path = Path.home() / ".local" / "share" / "KiCad" / version / "plugins"
    
    return docs_path

def install_plugin():
    """Install the OrthoRoute plugin"""
    
    # Source directory (where this script is)
    source_dir = Path(__file__).parent
    
    # Target directory
    plugins_dir = get_kicad_plugins_dir()
    target_dir = plugins_dir / "orthoroute_gpu"
    
    print(f"Installing OrthoRoute GPU Plugin...")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"KiCad Version: {find_kicad_version()}")
    
    # Create plugins directory if it doesn't exist
    plugins_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing installation
    if target_dir.exists():
        print(f"Removing existing installation...")
        shutil.rmtree(target_dir)
    
    # Create target directory
    target_dir.mkdir(exist_ok=True)
    
    # Files to copy
    files_to_copy = [
        "plugin.json",
        "orthoroute_gpu.py", 
        "icon.png",
        "README.md"
    ]
    
    # Copy files
    for filename in files_to_copy:
        source_file = source_dir / filename
        target_file = target_dir / filename
        
        if source_file.exists():
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied {filename}")
        else:
            print(f"❌ Missing {filename}")
            return False
    
    print(f"\n🎉 Installation completed successfully!")
    print(f"\nPlugin installed to: {target_dir}")
    print(f"\nNext steps:")
    print(f"1. Restart KiCad completely")
    print(f"2. Open PCB Editor")
    print(f"3. Look for 'OrthoRoute GPU' button in the toolbar")
    
    return True

def main():
    """Main installation function"""
    print("OrthoRoute GPU Plugin Installer")
    print("=" * 40)
    
    try:
        if install_plugin():
            print("\n✅ Installation successful!")
            return 0
        else:
            print("\n❌ Installation failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Installation error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
