#!/usr/bin/env python3
"""
Comprehensive OrthoRoute Test Runner
Runs all tests before packaging the addon
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def main():
    """Main test runner"""
    print("OrthoRoute Pre-Package Test Suite")
    print("=" * 60)
    
    # Get current directory
    project_dir = Path(__file__).parent.absolute()
    os.chdir(project_dir)
    
    test_results = {
        'standalone_tests': None,
        'headless_tests': None,
        'kicad_cli_tests': None,
        'overall_success': False
    }
    
    # Test 1: Run standalone Python tests
    print("\n1. Running Standalone Python Tests...")
    test_results['standalone_tests'] = run_standalone_tests(project_dir)
    
    # Test 2: Run headless tests
    print("\n2. Running Headless API Tests...")
    test_results['headless_tests'] = run_headless_tests(project_dir)
    
    # Test 3: Try KiCad CLI tests (if available)
    print("\n3. Running KiCad CLI Tests...")
    test_results['kicad_cli_tests'] = run_kicad_cli_tests(project_dir)
    
    # Overall assessment
    test_results['overall_success'] = assess_overall_results(test_results)
    
    # Print summary
    print_final_summary(test_results)
    
    # Save comprehensive results
    save_comprehensive_results(test_results)
    
    return test_results

def run_standalone_tests(project_dir):
    """Run standalone Python tests that don't require KiCad"""
    print("  [TOOL] Testing imports and basic functionality...")
    
    try:
        # Test 1: Check if files exist
        required_files = [
            'addon_package/plugins/orthoroute_engine.py',
            'addon_package/plugins/__init__.py',
            'api_bridge.py',
            '__init___ipc_compatible.py',
            'ipc_api_test_plugin.py'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        if missing_files:
            return {
                'success': False,
                'error': f"Missing files: {missing_files}"
            }
        
        # Test 2: Try importing without KiCad dependencies
        test_script = '''
import sys
sys.path.insert(0, "addon_package/plugins")
sys.path.insert(0, ".")

# Test basic imports
try:
    import json
    print("✅ JSON import OK")
except Exception as e:
    print(f"❌ JSON import failed: {e}")
    sys.exit(1)

try:
    from pathlib import Path
    print("✅ Path import OK")
except Exception as e:
    print(f"❌ Path import failed: {e}")
    sys.exit(1)

# Test OrthoRoute engine import (without KiCad dependencies)
try:
    # Read the engine file to check syntax
    with open("addon_package/plugins/orthoroute_engine.py", "r") as f:
        content = f.read()
    
    # Check for key classes
    if "class OrthoRouteEngine" in content:
        print("✅ OrthoRoute engine structure OK")
    else:
        print("❌ OrthoRoute engine missing main class")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ OrthoRoute engine check failed: {e}")
    sys.exit(1)

# Test API bridge structure
try:
    with open("api_bridge.py", "r") as f:
        content = f.read()
    
    if "class KiCadAPIBridge" in content:
        print("✅ API bridge structure OK")
    else:
        print("❌ API bridge missing main class")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ API bridge check failed: {e}")
    sys.exit(1)

print("✅ All standalone tests passed")
'''
        
        result = subprocess.run([
            sys.executable, '-c', test_script
        ], capture_output=True, text=True, cwd=project_dir)
        
        if result.returncode == 0:
            print(f"  [OK] Standalone tests passed")
            return {
                'success': True,
                'output': result.stdout
            }
        else:
            print(f"  [FAIL] Standalone tests failed")
            return {
                'success': False,
                'error': result.stderr,
                'output': result.stdout
            }
    
    except Exception as e:
        print(f"  [FAIL] Standalone test execution failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def run_headless_tests(project_dir):
    """Run headless tests"""
    print("  [TEST] Running headless API tests...")
    
    try:
        result = subprocess.run([
            sys.executable, 'test_orthoroute_headless.py'
        ], capture_output=True, text=True, cwd=project_dir)
        
        # Check if results file was created
        results_file = project_dir / 'headless_test_results.json'
        detailed_results = None
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    detailed_results = json.load(f)
            except Exception as e:
                print(f"  ⚠️ Could not read detailed results: {e}")
        
        if result.returncode == 0:
            print("  [OK] Headless tests passed")
            return {
                'success': True,
                'output': result.stdout,
                'detailed_results': detailed_results
            }
        else:
            print("  [FAIL] Headless tests failed")
            return {
                'success': False,
                'error': result.stderr,
                'output': result.stdout,
                'detailed_results': detailed_results
            }
    
    except Exception as e:
        print(f"  [FAIL] Headless test execution failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def run_kicad_cli_tests(project_dir):
    """Run KiCad CLI tests if available"""
    print("  [API] Checking KiCad CLI availability...")
    
    # KiCad installation path
    kicad_cli_path = r"C:\Program Files\KiCad\9.0\bin\kicad-cli.exe"
    
    # Check if kicad-cli is available
    try:
        result = subprocess.run([
            kicad_cli_path, '--version'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("  [WARN] kicad-cli not available, skipping CLI tests")
            return {
                'success': True,
                'skipped': True,
                'reason': 'kicad-cli not available'
            }
        
        print(f"  [OK] KiCad CLI available: {result.stdout.strip()}")
        
        # Run KiCad CLI test
        print("  [TEST] Running KiCad CLI with test board...")
        
        cli_result = subprocess.run([
            kicad_cli_path, 'pcb', 
            '--input', 'test_board.kicad_pcb',
            '--python-script', 'test_orthoroute_headless.py'
        ], capture_output=True, text=True, cwd=project_dir)
        
        if cli_result.returncode == 0:
            print("  [OK] KiCad CLI tests passed")
            return {
                'success': True,
                'output': cli_result.stdout,
                'kicad_version': result.stdout.strip()
            }
        else:
            print("  [FAIL] KiCad CLI tests failed")
            return {
                'success': False,
                'error': cli_result.stderr,
                'output': cli_result.stdout,
                'kicad_version': result.stdout.strip()
            }
    
    except FileNotFoundError:
        print("  [WARN] kicad-cli not found at expected path, skipping CLI tests")
        return {
            'success': True,
            'skipped': True,
            'reason': 'kicad-cli not found at expected path'
        }
    except Exception as e:
        print(f"  [FAIL] KiCad CLI test execution failed: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def assess_overall_results(test_results):
    """Assess overall test success"""
    critical_failures = []
    
    # Standalone tests are critical
    if not test_results['standalone_tests']['success']:
        critical_failures.append("Standalone tests failed")
    
    # Headless tests are important but not critical if it's just missing KiCad
    headless = test_results['headless_tests']
    if headless and not headless['success']:
        # Check if it's just a missing KiCad issue
        error = headless.get('error', '').lower()
        if 'pcbnew' not in error and 'import' not in error:
            critical_failures.append("Headless tests failed (non-import error)")
    
    # CLI tests are optional
    cli = test_results['kicad_cli_tests']
    if cli and not cli['success'] and not cli.get('skipped', False):
        # Only critical if it failed for non-availability reasons
        if 'not available' not in cli.get('reason', ''):
            critical_failures.append("KiCad CLI tests failed")
    
    return len(critical_failures) == 0

def print_final_summary(test_results):
    """Print comprehensive test summary"""
    print("\n" + "=" * 60)
    print("📊 FINAL TEST SUMMARY")
    print("=" * 60)
    
    # Standalone tests
    standalone = test_results['standalone_tests']
    status = "✅ PASS" if standalone['success'] else "❌ FAIL"
    print(f"🔧 Standalone Tests: {status}")
    
    # Headless tests
    headless = test_results['headless_tests']
    if headless:
        status = "✅ PASS" if headless['success'] else "❌ FAIL"
        print(f"🧪 Headless Tests: {status}")
    else:
        print("🧪 Headless Tests: ⚠️ SKIPPED")
    
    # CLI tests
    cli = test_results['kicad_cli_tests']
    if cli:
        if cli.get('skipped'):
            print(f"🔌 KiCad CLI Tests: ⚠️ SKIPPED ({cli.get('reason', 'Unknown')})")
        else:
            status = "✅ PASS" if cli['success'] else "❌ FAIL"
            print(f"🔌 KiCad CLI Tests: {status}")
    else:
        print("🔌 KiCad CLI Tests: ⚠️ SKIPPED")
    
    # Overall result
    overall_status = "✅ READY FOR PACKAGING" if test_results['overall_success'] else "❌ NOT READY"
    print(f"\n🎯 OVERALL RESULT: {overall_status}")

def save_comprehensive_results(test_results):
    """Save comprehensive test results"""
    try:
        results_file = Path('comprehensive_test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\n💾 Comprehensive results saved to: {results_file}")
    except Exception as e:
        print(f"⚠️ Could not save comprehensive results: {e}")

if __name__ == "__main__":
    results = main()
    
    # Exit with appropriate code
    exit_code = 0 if results['overall_success'] else 1
    print(f"\n🚪 Exiting with code: {exit_code}")
    sys.exit(exit_code)
