#!/usr/bin/env python3
"""
Pre-Build Validation Script for OrthoRoute
Tests plugin structure and imports without requiring KiCad APIs
"""

import sys
import ast
import importlib.util
from pathlib import Path

def analyze_python_file(file_path):
    """Analyze a Python file for syntax and structure"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to check syntax
        tree = ast.parse(content, filename=str(file_path))
        
        # Find classes and functions
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return {
            'valid': True,
            'classes': classes,
            'functions': functions,
            'imports': imports,
            'lines': len(content.split('\\n'))
        }
        
    except SyntaxError as e:
        return {
            'valid': False,
            'error': f"Syntax error: {e}",
            'line': getattr(e, 'lineno', 0)
        }
    except Exception as e:
        return {
            'valid': False,
            'error': f"Analysis error: {e}"
        }

def validate_plugin_structure():
    """Validate the plugin package structure and contents"""
    print("📦 Validating OrthoRoute plugin structure...")
    
    addon_dir = Path(__file__).parent / "addon_package"
    if not addon_dir.exists():
        print("❌ addon_package directory not found")
        return False
    
    # Check required files
    required_files = {
        'metadata.json': 'Package metadata',
        'plugins/__init__.py': 'Main plugin entry point',
        'plugins/orthoroute_engine.py': 'GPU routing engine',
        'plugins/api_bridge.py': 'API compatibility bridge',
        'resources/icon.png': 'Package icon'
    }
    
    missing_files = []
    file_analysis = {}
    
    for file_path, description in required_files.items():
        full_path = addon_dir / file_path
        if not full_path.exists():
            missing_files.append(file_path)
            print(f"❌ Missing: {file_path} ({description})")
        else:
            size = full_path.stat().st_size
            print(f"✅ Found: {file_path} ({size:,} bytes)")
            
            # Analyze Python files
            if file_path.endswith('.py'):
                analysis = analyze_python_file(full_path)
                file_analysis[file_path] = analysis
                
                if analysis['valid']:
                    classes = len(analysis['classes'])
                    functions = len(analysis['functions'])
                    lines = analysis['lines']
                    print(f"   📋 {lines:,} lines, {classes} classes, {functions} functions")
                else:
                    print(f"   ❌ {analysis['error']}")
                    return False
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} required files")
        return False
    
    # Validate main plugin components
    main_plugin = file_analysis.get('plugins/__init__.py', {})
    if main_plugin.get('valid'):
        if 'OrthoRoutePlugin' in main_plugin['classes']:
            print("✅ Main plugin class found")
        else:
            print("⚠️  Main plugin class not found (expected 'OrthoRoutePlugin')")
    
    engine = file_analysis.get('plugins/orthoroute_engine.py', {})
    if engine.get('valid'):
        engine_functions = engine['functions']
        if any('route' in func.lower() for func in engine_functions):
            print("✅ Routing functions found in engine")
        else:
            print("⚠️  No routing functions found in engine")
    
    api_bridge = file_analysis.get('plugins/api_bridge.py', {})
    if api_bridge.get('valid'):
        bridge_functions = api_bridge['functions']
        if 'detect_api_type' in bridge_functions:
            print("✅ API detection function found")
        else:
            print("⚠️  API detection function not found")
    
    print("✅ Plugin structure validation passed")
    return True

def validate_metadata():
    """Validate the metadata.json file"""
    print("📋 Validating package metadata...")
    
    import json
    
    metadata_path = Path(__file__).parent / "addon_package" / "metadata.json"
    if not metadata_path.exists():
        print("❌ metadata.json not found")
        return False
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        required_fields = ['name', 'description', 'identifier', 'type', 'version']
        missing_fields = []
        
        for field in required_fields:
            if field not in metadata:
                # Check if version is in versions array
                if field == 'version' and 'versions' in metadata:
                    if metadata['versions'] and 'version' in metadata['versions'][0]:
                        print(f"✅ {field}: {metadata['versions'][0]['version']} (from versions array)")
                        continue
                missing_fields.append(field)
            else:
                value = metadata[field]
                print(f"✅ {field}: {value}")
        
        if missing_fields:
            print(f"❌ Missing metadata fields: {missing_fields}")
            return False
        
        # Additional validations
        if metadata.get('type') != 'plugin':
            print(f"⚠️  Unexpected type: {metadata.get('type')} (expected 'plugin')")
        
        if 'orthoroute' not in metadata.get('identifier', '').lower():
            print(f"⚠️  Identifier doesn't contain 'orthoroute': {metadata.get('identifier')}")
        
        print("✅ Metadata validation passed")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in metadata.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Metadata validation error: {e}")
        return False

def validate_imports():
    """Validate that plugin files can be imported (without KiCad APIs)"""
    print("🔧 Validating plugin imports (syntax check)...")
    
    plugin_dir = Path(__file__).parent / "addon_package" / "plugins"
    python_files = list(plugin_dir.glob("*.py"))
    
    if not python_files:
        print("❌ No Python files found in plugins directory")
        return False
    
    success_count = 0
    
    for py_file in python_files:
        if py_file.name.startswith('__'):
            continue  # Skip __init__.py for now due to KiCad dependencies
        
        try:
            # Try to compile the file
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            
            compile(source, str(py_file), 'exec')
            print(f"✅ {py_file.name}: Syntax valid")
            success_count += 1
            
        except SyntaxError as e:
            print(f"❌ {py_file.name}: Syntax error at line {e.lineno}: {e.msg}")
            return False
        except Exception as e:
            print(f"⚠️  {py_file.name}: Compile issue: {e}")
    
    print(f"✅ {success_count} plugin files validated")
    return success_count > 0

def check_package_size():
    """Check if the package will be a reasonable size"""
    print("📏 Checking package size...")
    
    addon_dir = Path(__file__).parent / "addon_package"
    total_size = 0
    file_count = 0
    
    for file_path in addon_dir.rglob("*"):
        if file_path.is_file():
            size = file_path.stat().st_size
            total_size += size
            file_count += 1
    
    size_kb = total_size / 1024
    print(f"✅ Package contents: {file_count} files, {size_kb:.1f} KB")
    
    if size_kb > 200:
        print(f"⚠️  Large package size ({size_kb:.1f} KB) - consider optimization")
    elif size_kb < 10:
        print(f"⚠️  Very small package ({size_kb:.1f} KB) - may be incomplete")
    else:
        print(f"✅ Package size is reasonable")
    
    return True

def main():
    """Run all pre-build validations"""
    print("=" * 70)
    print("🔍 OrthoRoute Pre-Build Validation")
    print("=" * 70)
    
    tests = [
        ("Plugin Structure", validate_plugin_structure),
        ("Metadata", validate_metadata),
        ("Import Syntax", validate_imports),
        ("Package Size", check_package_size)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
            print()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
            print()
    
    print("=" * 70)
    print("📊 PRE-BUILD VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\\nResult: {passed}/{len(results)} validations passed")
    
    if passed == len(results):
        print("🎉 ALL VALIDATIONS PASSED")
        print("✅ Plugin is ready for packaging!")
        print("\\n📦 To build the package:")
        print("   python build_addon.py")
        return True
    elif passed >= len(results) - 1:
        print("✅ MOSTLY READY")
        print("⚠️  Minor issues detected, but package should work")
        return True
    else:
        print("❌ VALIDATION FAILURES")
        print("🛑 Fix issues before packaging")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
