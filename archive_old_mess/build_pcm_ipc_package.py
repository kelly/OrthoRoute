#!/usr/bin/env python3
"""
Build script to create PCM package with IPC runtime support
This creates a ZIP file that can be installed via KiCad's Plugin and Content Manager
"""

import os
import zipfile
import shutil
from pathlib import Path

def create_pcm_ipc_package():
    """Create PCM package with IPC runtime"""
    print("🏗️  Building PCM IPC Package...")
    
    # Package source directory
    pcm_dir = Path("pcm_package")
    
    # Output zip file
    output_zip = "ultra-simple-ipc-pcm-package.zip"
    
    # Remove existing zip if it exists
    if os.path.exists(output_zip):
        os.remove(output_zip)
        print(f"🗑️  Removed existing {output_zip}")
    
    # Create zip file with proper PCM structure
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add metadata.json at root
        metadata_path = pcm_dir / "metadata.json"
        if metadata_path.exists():
            zipf.write(metadata_path, "metadata.json")
            print("✅ Added metadata.json")
        
        # Add plugins directory
        plugins_dir = pcm_dir / "plugins"
        if plugins_dir.exists():
            for file_path in plugins_dir.rglob('*'):
                if file_path.is_file():
                    arcname = f"plugins/{file_path.relative_to(plugins_dir)}"
                    zipf.write(file_path, arcname)
                    print(f"✅ Added {arcname}")
        
        # Add resources directory (placeholder for now)
        resources_dir = pcm_dir / "resources"
        if resources_dir.exists():
            for file_path in resources_dir.rglob('*'):
                if file_path.is_file() and not file_path.name.endswith('.placeholder'):
                    arcname = f"resources/{file_path.relative_to(resources_dir)}"
                    zipf.write(file_path, arcname)
                    print(f"✅ Added {arcname}")
    
    print(f"✅ Created {output_zip}")
    print(f"📦 Package size: {os.path.getsize(output_zip)} bytes")
    
    # Verify zip contents
    print("\n📋 Package contents:")
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        for info in zipf.infolist():
            print(f"   {info.filename}")
    
    print(f"\n🎉 PCM IPC Package ready: {output_zip}")
    print("📥 Install via: KiCad → Plugin and Content Manager → Install from file...")
    
    return output_zip

if __name__ == "__main__":
    create_pcm_ipc_package()
