"""
Development installation script for OrthoRoute KiCad addon
Installs directly to KiCad's plugin directory for quick testing
"""

import os
import shutil
from pathlib import Path

def install_for_development():
    """Install the addon directly to KiCad for development testing"""
    
    # Paths
    addon_dir = Path(__file__).parent / "addon_package"
    kicad_plugins_dir = Path("c:/Users/Benchoff/Documents/KiCad/9.0/scripting/plugins")
    target_dir = kicad_plugins_dir / "OrthoRoute"
    
    # Create KiCad plugins directory if it doesn't exist
    kicad_plugins_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove existing installation
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f"Removed existing installation: {target_dir}")
    
    # Create target directory
    target_dir.mkdir()
    
    # Copy plugin files
    plugins_src = addon_dir / "plugins"
    if plugins_src.exists():
        for file in plugins_src.iterdir():
            if file.is_file():
                shutil.copy2(file, target_dir)
                print(f"Copied: {file.name}")
    
    # Copy resources (icons, etc.)
    resources_src = addon_dir / "resources"
    if resources_src.exists():
        resources_target = target_dir / "resources"
        resources_target.mkdir(exist_ok=True)
        for file in resources_src.iterdir():
            if file.is_file():
                shutil.copy2(file, resources_target)
                print(f"Copied resource: {file.name}")
    
    print(f"\n✅ Development installation complete!")
    print(f"Plugin installed to: {target_dir}")
    print(f"\nTo test:")
    print(f"1. Restart KiCad completely")
    print(f"2. Look for 'OrthoRoute GPU Autorouter' in Tools menu")
    print(f"3. Or check the toolbar for the routing icon")
    
    # Show what was installed
    print(f"\nInstalled files:")
    for file in target_dir.rglob("*"):
        if file.is_file():
            rel_path = file.relative_to(target_dir)
            print(f"  {rel_path}")

def uninstall_development():
    """Remove the development installation"""
    target_dir = Path("c:/Users/Benchoff/Documents/KiCad/9.0/scripting/plugins/OrthoRoute")
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
        print(f"✅ Removed development installation: {target_dir}")
    else:
        print(f"No development installation found at: {target_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "uninstall":
        uninstall_development()
    else:
        install_for_development()
