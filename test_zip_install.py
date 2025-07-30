#!/usr/bin/env python3
"""
Test that the zip file is properly structured for KiCad Plugin Manager installation
"""

import os
import zipfile
import json

def test_zip_structure():
    """Test that the zip file has the correct structure for KiCad"""
    
    zip_path = "orthoroute-kicad-addon.zip"
    
    if not os.path.exists(zip_path):
        print("❌ Zip file not found. Run 'python build_addon.py' first.")
        return False
    
    print("🔍 Testing KiCad plugin zip structure...")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        files = zf.namelist()
        
        # Required files for KiCad plugin
        required_files = [
            'metadata.json',
            'plugins/__init__.py'
        ]
        
        print(f"📦 Zip contains {len(files)} files")
        
        # Check required files
        missing = []
        for req in required_files:
            if req not in files:
                missing.append(req)
            else:
                print(f"✅ {req}")
        
        if missing:
            print(f"❌ Missing required files: {missing}")
            return False
        
        # Check metadata.json structure
        try:
            metadata_content = zf.read('metadata.json').decode('utf-8')
            metadata = json.loads(metadata_content)
            
            required_metadata = ['name', 'description', 'identifier', 'type', 'version']
            for field in required_metadata:
                if field in metadata:
                    print(f"✅ metadata.{field}: {metadata[field]}")
                else:
                    print(f"❌ Missing metadata field: {field}")
                    return False
                    
        except Exception as e:
            print(f"❌ Invalid metadata.json: {e}")
            return False
        
        # Check plugin structure
        plugin_files = [f for f in files if f.startswith('plugins/') and f.endswith('.py')]
        print(f"✅ Found {len(plugin_files)} Python plugin files")
        
        # Check icons
        icon_files = [f for f in files if f.endswith('.png')]
        print(f"✅ Found {len(icon_files)} icon files")
        
        # Check size
        size_kb = os.path.getsize(zip_path) / 1024
        print(f"✅ Package size: {size_kb:.1f} KB")
        
        if size_kb > 100:
            print("⚠️  Large package size - consider optimization")
        
    print("\n🎉 Zip file structure is correct for KiCad Plugin Manager!")
    return True

def main():
    print("=" * 60)
    print("📦 KiCad Plugin Zip Structure Test")
    print("=" * 60)
    
    if test_zip_structure():
        print("\n✅ READY FOR INSTALLATION")
        print("\nTo install:")
        print("1. Open KiCad PCB Editor")
        print("2. Tools → Plugin and Content Manager")
        print("3. Install from File → Select orthoroute-kicad-addon.zip")
        print("4. Restart KiCad")
        print("5. Look for 'OrthoRoute GPU Autorouter' in Tools → External Plugins")
        return True
    else:
        print("\n❌ ZIP STRUCTURE ISSUES")
        print("Fix the issues above before attempting installation.")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
