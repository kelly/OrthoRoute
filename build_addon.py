"""
Build script for OrthoRoute KiCad addon package
Creates a zip file suitable for KiCad Plugin and Content Manager
"""

import os
import zipfile
import json
from pathlib import Path

def create_addon_package():
    """Create the addon package zip file"""
    
    # Paths
    addon_dir = Path(__file__).parent / "addon_package"
    output_file = Path(__file__).parent / "orthoroute-kicad-addon.zip"
    
    # Remove existing package
    if output_file.exists():
        output_file.unlink()
    
    print("Creating OrthoRoute KiCad addon package...")
    
    # Create zip file
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from addon_package directory
        for root, dirs, files in os.walk(addon_dir):
            for file in files:
                file_path = Path(root) / file
                # Calculate relative path from addon_package
                arc_path = file_path.relative_to(addon_dir)
                zipf.write(file_path, arc_path)
                print(f"  Added: {arc_path}")
    
    # Verify the package structure
    print(f"\nPackage created: {output_file}")
    print(f"Package size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Show contents
    print("\nPackage contents:")
    with zipfile.ZipFile(output_file, 'r') as zipf:
        for info in zipf.infolist():
            print(f"  {info.filename} ({info.file_size} bytes)")
    
    # Validate metadata
    with zipfile.ZipFile(output_file, 'r') as zipf:
        try:
            metadata_content = zipf.read('metadata.json')
            metadata = json.loads(metadata_content)
            print(f"\nMetadata validation:")
            print(f"  Name: {metadata['name']}")
            print(f"  Version: {metadata['versions'][0]['version']}")
            print(f"  Identifier: {metadata['identifier']}")
            print(f"  Type: {metadata['type']}")
            print("  ✓ Metadata is valid JSON")
        except Exception as e:
            print(f"  ✗ Metadata validation failed: {e}")
            return False
    
    print(f"\n✅ Addon package created successfully: {output_file}")
    print(f"\nTo install:")
    print(f"1. Open KiCad")
    print(f"2. Go to Tools → Plugin and Content Manager")
    print(f"3. Click 'Install from File'")
    print(f"4. Select: {output_file}")
    
    return True

if __name__ == "__main__":
    create_addon_package()
