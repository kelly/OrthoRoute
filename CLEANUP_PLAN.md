# OrthoRoute Project Cleanup Plan

## Current Issues
- Multiple overlapping implementations
- Empty and unused files throughout project
- Unclear project structure with 15+ directories
- Development code mixed with production code
- Archive directories competing with current code

## Cleanup Strategy

### 1. Archive Everything Non-Essential
Move all development/debug/test files to a single archive:
- All `debug_*.py` files 
- All `test_*.py` files in root
- All experimental directories (`pcm_*`, `orthoroute_*`, `ultra_simple_*`, etc.)
- All old documentation files
- All build artifacts and ZIP files

### 2. Create Clean Structure
```
OrthoRoute/
├── src/                    # Core plugin source code
│   ├── __init__.py        # Main plugin entry point
│   ├── routing_engine.py  # GPU routing engine  
│   ├── ui_dialog.py       # Configuration UI
│   └── utils.py           # Helper functions
├── assets/                # Icons and images
├── docs/                  # User documentation
├── tests/                 # Essential tests only
├── build/                 # Build outputs
├── archive/               # All old development code
├── README.md              # Main documentation
├── LICENSE                # License file
├── build.py               # Single build script
└── setup.py               # Installation script
```

### 3. Consolidate Code
- Keep only the working plugin implementation
- Merge scattered utility functions
- Remove all empty files
- Clean up import statements

### 4. Archive Targets
Files/directories to archive:
- `debug_*.py` (all debug scripts)
- `*_test_*.py` (scattered test files)
- `pcm_package/`, `pcm_cpp_package/` 
- `orthoroute_gpu_package/`, `orthoroute_ipc_correct/`, `orthoroute_native_ipc/`
- `ultra_simple_ipc_plugin/`, `native_ipc_plugin/`
- `addon_package/` (move to archive, extract working code to src/)
- All `.zip` files
- All markdown files except README.md and core docs
- `cleanup_temp/`, `cleanup_project.py`
- All `build_*.py` scripts except main build script

## Result
A clean, professional project structure with:
- Single source directory with working code
- Clear build and installation process  
- Comprehensive documentation
- All experimental code safely archived
- Easy to understand and contribute to
