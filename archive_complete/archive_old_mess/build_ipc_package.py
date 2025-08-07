"""
Build script for KiCad 9.0+ IPC Plugin Package
Creates correctly structured ZIP for Plugin and Content Manager
"""

import os
import sys
import zipfile
import json
import hashlib
from pathlib import Path

def calculate_sha256(file_path):
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()

def get_directory_size(path):
    """Get total size of directory in bytes"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total_size += os.path.getsize(filepath)
    return total_size

def build_ipc_package():
    """Build the IPC plugin package"""
    
    # Paths
    script_dir = Path(__file__).parent
    package_dir = script_dir / "orthoroute_ipc_correct"
    output_zip = script_dir / "orthoroute-gpu-1.0.0.zip"
    
    if not package_dir.exists():
        print(f"Error: Package directory not found: {package_dir}")
        return False
        
    # Verify required structure
    required_files = [
        package_dir / "metadata.json",
        package_dir / "plugins" / "plugin.json",
        package_dir / "plugins" / "gpu_autorouter.py",
        package_dir / "plugins" / "requirements.txt"
    ]
    
    for required_file in required_files:
        if not required_file.exists():
            print(f"Error: Required file missing: {required_file}")
            return False
            
    # Remove existing zip
    if output_zip.exists():
        output_zip.unlink()
        print(f"Removed existing {output_zip.name}")
        
    # Create zip file with correct structure
    print(f"Creating KiCad 9.0+ IPC Plugin Package: {output_zip.name}")
    print("Structure:")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add files from package directory
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                # Calculate path relative to package directory (this becomes the ZIP root)
                arc_path = file_path.relative_to(package_dir)
                zipf.write(file_path, arc_path)
                print(f"  ✅ {arc_path}")
    
    # Verify ZIP structure
    print(f"\nVerifying ZIP structure...")
    with zipfile.ZipFile(output_zip, 'r') as zipf:
        files_in_zip = zipf.namelist()
        
        # Check for required root-level files
        if "metadata.json" not in files_in_zip:
            print("❌ Error: metadata.json not at root level!")
            return False
            
        if "plugins/plugin.json" not in files_in_zip:
            print("❌ Error: plugins/plugin.json missing!")
            return False
            
        if "plugins/gpu_autorouter.py" not in files_in_zip:
            print("❌ Error: plugins/gpu_autorouter.py missing!")
            return False
            
        print("✅ ZIP structure verified!")
    
    # Calculate package info
    zip_size = output_zip.stat().st_size
    zip_hash = calculate_sha256(output_zip)
    install_size = get_directory_size(package_dir)
    
    print(f"\n🎉 KiCad IPC Plugin Package created successfully!")
    print(f"📦 File: {output_zip}")
    print(f"📏 Size: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")
    print(f"🔒 SHA256: {zip_hash}")
    print(f"💾 Install size: {install_size:,} bytes")
    
    # Validate metadata.json
    try:
        with open(package_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        print(f"✅ metadata.json is valid JSON")
        print(f"   Plugin: {metadata['name']}")
        print(f"   Version: {metadata['versions'][0]['version']}")
        print(f"   Runtime: {metadata['versions'][0]['runtime']}")
    except Exception as e:
        print(f"❌ metadata.json validation failed: {e}")
        return False
    
    # Validate plugin.json
    try:
        with open(package_dir / "plugins" / "plugin.json", 'r') as f:
            plugin_config = json.load(f)
        print(f"✅ plugin.json is valid JSON")
        print(f"   Actions: {len(plugin_config['actions'])}")
        print(f"   Runtime: {plugin_config['runtime']['type']}")
    except Exception as e:
        print(f"❌ plugin.json validation failed: {e}")
        return False
    
    return True

def main():
    """Main build function"""
    print("🚀 Building OrthoRoute GPU - KiCad 9.0+ IPC Plugin")
    print("=" * 60)
    
    if build_ipc_package():
        print(f"\n✅ Build completed successfully!")
        print(f"\n📋 Installation Instructions:")
        print(f"1. Open KiCad PCB Editor")
        print(f"2. Go to Tools → Plugin and Content Manager")
        print(f"3. Click 'Install from File'")
        print(f"4. Select: orthoroute-gpu-1.0.0.zip")
        print(f"5. Enable API server: Preferences → Plugins → 'Enable external plugin API server'")
        print(f"6. Restart KiCad")
        print(f"7. Look for 'Run GPU Autorouter' button in PCB Editor toolbar! 🎯")
        print(f"\n🎉 This uses the correct KiCad 9.0+ IPC structure with runtime: 'ipc'")
    else:
        print(f"\n❌ Build failed!")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
