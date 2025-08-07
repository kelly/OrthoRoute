#!/usr/bin/env python3
"""
Cleanup Script - Archive old project mess and keep only clean structure
"""

import os
import shutil
from pathlib import Path

def main():
    """Clean up the project directory"""
    
    root = Path(".")
    archive_dir = root / "archive_old_mess"
    
    # Create archive directory
    archive_dir.mkdir(exist_ok=True)
    
    print("🧹 Cleaning up OrthoRoute project directory...")
    print("=" * 50)
    
    # Files and directories to archive (the old mess)
    to_archive = [
        # Old build scripts
        "build_addon.py", "build_gpu_package.py", "build_ipc_package.py", 
        "build_minimal.py", "build_pcm_ipc_package.py", "install_dev.py",
        
        # Old plugin files
        "bare_minimum_plugin.json", "bare_minimum_working.py", "bulletproof_minimal.py",
        "corrected_minimal_plugin.py", "final_working_plugin.py", "hybrid_test_plugin.py",
        "minimal_test_plugin.json", "orthoroute_ipc_plugin.py", "orthoroute_standalone.py",
        "orthoroute_standalone_gui.py", "plugin.json", "simple_test_plugin.py", 
        "ultra_minimal.py", "validate_plugin.py",
        
        # Old test files  
        "debug_board_api.py", "debug_commit.py", "debug_ipc_setup.py", "debug_layers.py",
        "debug_plugin_direct.py", "debug_python_test.py", "debug_track_creation.py",
        "debug_vector2.py", "debug_vector2_methods.py", "ipc_api_test.py", 
        "kicad_python_env_check.py", "minimal_ipc_test.py", "python_env_check.py",
        "test_continuous_server.py", "test_crash_fix.py", "test_crash_fix_validation.py",
        "test_ipc_api.py", "test_ipc_connection.py", "test_ipc_installation.py",
        "test_ipc_only.py", "test_ipc_plugin.py", "test_kicad_env.py", "test_minimal.py",
        "test_minimal_directly.py", "test_package_summary.py", "test_routing_request.json",
        "test_server.py", "test_server_standalone.py", "test_simple_server.py", 
        "test_track_fix.py",
        
        # Old ZIP files
        "minimal-track-test.zip", "orthoroute-complete-pcm-package.zip", 
        "orthoroute-fixed-plugin.zip", "orthoroute-gpu-1.0.0.zip", "orthoroute-gpu-package.zip",
        "orthoroute-hybrid.zip", "orthoroute-ipc-only.zip", "orthoroute-kicad-addon.zip",
        "ultra-simple-ipc-corrected.zip", "ultra-simple-ipc-pcm-package.zip", "ultra-simple-ipc.zip",
        
        # Old directories
        "addon_package", "development", "minimal_plugin_package", "minimal_test_package",
        "native_ipc_plugin", "orthoroute_gpu_package", "orthoroute_ipc_correct", 
        "orthoroute_native_ipc", "pcm_cpp_package", "pcm_package", "test_robust",
        "test_work", "test_work_dir", "ultra_simple_ipc_plugin",
        
        # Old documentation files
        "COMPLETE_PROJECT_ANALYSIS.md", "CRASH_ROOT_CAUSE_ANALYSIS.md", "IPC_ONLY_MIGRATION_COMPLETE.md",
        "IPC_PLUGIN_READY.md", "IPC_PLUGIN_SOLUTION.md", "IPC_PLUGIN_VISIBILITY_DEBUG.md",
        "IPC_STRUCTURE_FIX.md", "KICAD_CRASH_FIX_IMPLEMENTATION.md", "KICAD_EXIT_CRASH_FIX.md",
        "KICAD_IPC_API_FIXES.md", "KICAD_PLUGIN_ISSUES_SOLVED.md", "MANUAL_IPC_INSTALLATION.md",
        "MINIMAL_APPROACH.md", "MODERN_KICAD_DEVELOPMENT.md", "PLUGIN_DEBUGGING_CHECKLIST.md",
        "SUBPROCESS_SOLUTION.md"
    ]
    
    # Keep these important directories/files
    keep = {
        ".git", ".gitignore", ".gitattributes", "README.md", "INSTALL.md",
        "src", "build", "assets", "docs", "tests", ".venv",
        "build_plugin.py", "README_CLEAN.md", "cleanup_temp"
    }
    
    archived_count = 0
    
    for item_name in to_archive:
        item_path = root / item_name
        if item_path.exists():
            target_path = archive_dir / item_name
            
            try:
                if item_path.is_dir():
                    shutil.move(str(item_path), str(target_path))
                    print(f"  📁 Moved directory: {item_name}")
                else:
                    shutil.move(str(item_path), str(target_path))
                    print(f"  📄 Moved file: {item_name}")
                
                archived_count += 1
                
            except Exception as e:
                print(f"  ❌ Error moving {item_name}: {e}")
    
    print(f"\n✅ Cleanup completed!")
    print(f"📦 Archived {archived_count} items to: {archive_dir}")
    
    print("\n📁 Clean project structure:")
    print("  src/           - Plugin source code")
    print("  build/         - Build output") 
    print("  assets/        - Icons and images")
    print("  docs/          - Documentation")
    print("  tests/         - Test files")
    print("  build_plugin.py - Build script")
    print(f"  {archive_dir.name}/ - Old project files")
    
    print(f"\n🎯 Ready for development with clean structure!")

if __name__ == "__main__":
    main()
