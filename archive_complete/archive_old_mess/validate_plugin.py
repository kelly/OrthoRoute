#!/usr/bin/env python3
"""
Validate plugin.json structure for KiCad 9.0 PCM compatibility
"""

import json
import os

def validate_plugin_json():
    plugin_path = "addon_package/plugin.json"
    
    print("🔍 Validating plugin.json structure...")
    
    with open(plugin_path, 'r') as f:
        plugin_data = json.load(f)
    
    print(f"✅ JSON is valid")
    print(f"📋 Plugin name: {plugin_data.get('name')}")
    print(f"🆔 Identifier: {plugin_data.get('identifier')}")
    print(f"📦 Type: {plugin_data.get('type')}")
    
    # Check actions
    actions = plugin_data.get('actions', [])
    print(f"🎯 Actions count: {len(actions)}")
    
    for i, action in enumerate(actions):
        print(f"  Action {i+1}:")
        print(f"    - ID: {action.get('identifier')}")
        print(f"    - Name: {action.get('name')}")
        print(f"    - Entrypoint: {action.get('entrypoint')}")
        print(f"    - Show in toolbar: {action.get('show_in_toolbar')}")
        print(f"    - Show in menu: {action.get('show_in_menu')}")
        
        # Check if entrypoint file exists
        entrypoint = action.get('entrypoint')
        if entrypoint:
            entrypoint_path = os.path.join('addon_package', entrypoint)
            if os.path.exists(entrypoint_path):
                print(f"    ✅ Entrypoint file exists: {entrypoint}")
            else:
                print(f"    ❌ Entrypoint file missing: {entrypoint}")
    
    # Check icon file
    icon_path = "addon_package/resources/icon.png"
    if os.path.exists(icon_path):
        print(f"✅ Icon file exists")
    else:
        print(f"❌ Icon file missing")
    
    print("\n🧪 Testing entrypoint imports...")
    
    # Test if we can import the entrypoint files
    import sys
    sys.path.insert(0, 'addon_package')
    
    for action in actions:
        entrypoint = action.get('entrypoint')
        if entrypoint and entrypoint.endswith('.py'):
            module_name = entrypoint.replace('.py', '').replace('/', '.')
            try:
                # Don't actually import, just check syntax
                with open(f'addon_package/{entrypoint}', 'r') as f:
                    code = f.read()
                compile(code, entrypoint, 'exec')
                print(f"    ✅ {entrypoint} syntax OK")
            except Exception as e:
                print(f"    ❌ {entrypoint} has issues: {e}")

if __name__ == "__main__":
    validate_plugin_json()
