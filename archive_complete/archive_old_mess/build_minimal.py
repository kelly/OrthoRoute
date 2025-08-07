#!/usr/bin/env python3
"""
Build script for minimal track test plugin
"""

import zipfile
import os
from pathlib import Path

def build_minimal_package():
    """Build the minimal test package"""
    
    print("🚀 Building Minimal Track Test Plugin")
    print("=" * 40)
    
    package_dir = Path("minimal_test_package")
    output_file = "minimal-track-test.zip"
    
    # Files to include
    files_to_include = [
        "plugin.json",
        "metadata.json", 
        "minimal_track_test.py",
        "README.md"
    ]
    
    # Create zip file
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_name in files_to_include:
            file_path = package_dir / file_name
            if file_path.exists():
                zipf.write(file_path, file_name)
                print(f"  ✅ Added: {file_name}")
            else:
                print(f"  ❌ Missing: {file_name}")
    
    # Get file size
    file_size = os.path.getsize(output_file)
    size_kb = file_size / 1024
    
    print(f"\n🎉 Package created: {output_file}")
    print(f"📦 Size: {size_kb:.1f} KB ({file_size} bytes)")
    
    print(f"\n📋 Package contents:")
    with zipfile.ZipFile(output_file, 'r') as zipf:
        for info in zipf.infolist():
            print(f"  📄 {info.filename} ({info.file_size} bytes)")
    
    print(f"\n🚀 Installation:")
    print(f"1. Open KiCad PCB Editor")
    print(f"2. Go to Tools → Plugin and Content Manager")
    print(f"3. Click 'Install from File'")
    print(f"4. Select: {output_file}")
    print(f"5. Restart KiCad")
    print(f"6. Look for 'Minimal Track Test' in Tools → External Plugins")

if __name__ == "__main__":
    build_minimal_package()
