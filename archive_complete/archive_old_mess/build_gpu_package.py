"""
Build script for OrthoRoute GPU PCM package
Creates a zip file ready for KiCad Plugin Manager installation
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

def build_gpu_package():
    """Build the GPU package zip file"""
    
    # Paths
    script_dir = Path(__file__).parent
    package_dir = script_dir / "orthoroute_gpu_package"
    output_zip = script_dir / "orthoroute-gpu-package.zip"
    
    if not package_dir.exists():
        print(f"Error: Package directory not found: {package_dir}")
        return False
        
    # Remove existing zip
    if output_zip.exists():
        output_zip.unlink()
        print(f"Removed existing {output_zip.name}")
        
    # Create zip file
    print(f"Creating {output_zip.name}...")
    
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(package_dir):
            for file in files:
                file_path = Path(root) / file
                arc_path = file_path.relative_to(package_dir)
                zipf.write(file_path, arc_path)
                print(f"  Added: {arc_path}")
    
    # Calculate package info
    zip_size = output_zip.stat().st_size
    zip_hash = calculate_sha256(output_zip)
    install_size = get_directory_size(package_dir)
    
    print(f"\nPackage created successfully!")
    print(f"File: {output_zip}")
    print(f"Size: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")
    print(f"SHA256: {zip_hash}")
    print(f"Install size: {install_size:,} bytes")
    
    # Update metadata with actual values
    metadata_file = package_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        # Update version info
        if metadata.get('versions'):
            version = metadata['versions'][0]
            version['download_size'] = zip_size
            version['download_sha256'] = zip_hash
            version['install_size'] = install_size
            
            # Set download URL (would be actual URL in production)
            version['download_url'] = f"https://github.com/bbenchoff/OrthoRoute/releases/download/v{version['version']}/orthoroute-gpu-package.zip"
            
        # Write updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"\nUpdated metadata.json with package info")
        
        # Recreate zip with updated metadata
        output_zip.unlink()
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(package_dir):
                for file in files:
                    file_path = Path(root) / file
                    arc_path = file_path.relative_to(package_dir)
                    zipf.write(file_path, arc_path)
        
        # Recalculate final hash
        final_hash = calculate_sha256(output_zip)
        final_size = output_zip.stat().st_size
        
        print(f"\nFinal package:")
        print(f"Size: {final_size:,} bytes")
        print(f"SHA256: {final_hash}")
    
    return True

def main():
    """Main build function"""
    print("Building OrthoRoute GPU PCM Package...")
    print("=" * 50)
    
    if build_gpu_package():
        print("\n✅ Build completed successfully!")
        print("\nTo install:")
        print("1. Open KiCad PCB Editor")
        print("2. Go to Tools → Plugin and Content Manager")
        print("3. Click 'Install from File'")
        print("4. Select orthoroute-gpu-package.zip")
    else:
        print("\n❌ Build failed!")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
