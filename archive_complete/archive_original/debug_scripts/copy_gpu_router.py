#!/usr/bin/env python3
"""
Copy the GPU router script to the plugin directory for packaging
"""

import shutil
import os

def copy_gpu_router():
    """Copy gpu_router_isolated.py to the addon_package/plugins/ directory"""
    
    # Source and destination paths
    source = 'gpu_router_isolated.py'
    dest_dir = 'addon_package/plugins/'
    dest = os.path.join(dest_dir, 'gpu_router_isolated.py')
    
    try:
        if os.path.exists(source):
            # Ensure destination directory exists
            os.makedirs(dest_dir, exist_ok=True)
            
            # Copy the file
            shutil.copy2(source, dest)
            print(f"✅ Copied {source} to {dest}")
            return True
        else:
            print(f"❌ Source file not found: {source}")
            return False
    except Exception as e:
        print(f"❌ Error copying file: {e}")
        return False

if __name__ == "__main__":
    copy_gpu_router()
