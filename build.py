#!/usr/bin/env python3
"""
OrthoRoute Build System - Unified builder for all package formats
Creates KiCad plugin packages in multiple formats for distribution
"""

import os
import sys
import json
import shutil
import zipfile
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrthoRouteBuildSystem:
    """Unified build system for OrthoRoute plugin packages"""
    
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent
        self.build_dir = self.project_root / "build"
        self.version = "1.0.0"
        
        # Package configurations
        self.packages = {
            'production': {
                'name': 'orthoroute-production',
                'description': 'OrthoRoute Professional PCB Autorouting Plugin',
                'main_file': 'src/orthoroute.py',
                'include_gpu': True,
                'include_docs': True,
                'include_tests': False,
                'include_debug': False
            },
            'lite': {
                'name': 'orthoroute',
                'description': 'OrthoRoute - Professional PCB autorouting functionality',
                'main_file': 'src/orthoroute.py',
                'include_gpu': False,
                'include_docs': False,
                'include_tests': False,
                'include_debug': False
            },
            'development': {
                'name': 'orthoroute-dev',
                'description': 'OrthoRoute Development Build with debugging tools',
                'main_file': 'src/orthoroute.py',
                'include_gpu': True,
                'include_docs': True,
                'include_tests': True,
                'include_debug': True
            }
        }
    
    def clean_build_directory(self):
        """Clean the build directory"""
        logger.info("🧹 Cleaning build directory...")
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        self.build_dir.mkdir(exist_ok=True)
        logger.info(f"✓ Build directory cleaned: {self.build_dir}")
    
    def create_plugin_metadata(self, package_config: Dict) -> Dict:
        """Create plugin metadata for KiCad"""
        return {
            "name": package_config['name'],
            "description": package_config['description'],
            "description_full": f"{package_config['description']} - GPU-accelerated PCB autorouting with real-time visualization",
            "identifier": package_config['name'].replace('-', '_'),
            "type": "plugin",
            "author": {
                "name": "OrthoRoute Team",
                "contact": {
                    "github": "https://github.com/bbenchoff/OrthoRoute"
                }
            },
            "maintainer": {
                "name": "OrthoRoute Team",
                "contact": {
                    "github": "https://github.com/bbenchoff/OrthoRoute"
                }
            },
            "license": "MIT",
            "resources": {
                "homepage": "https://github.com/bbenchoff/OrthoRoute"
            },
            "tags": [
                "pcb",
                "routing",
                "autorouting", 
                "gpu",
                "automation",
                "visualization"
            ],
            "keep_on_update": [],
            "versions": [
                {
                    "version": self.version,
                    "status": "stable",
                    "kicad_version": "9.0",
                    "kicad_version_max": "9.99",
                    "platforms": ["windows", "macos", "linux"],
                    "python_requires": ">=3.8",
                    "install_size": 0,  # Will be calculated
                    "download_sha256": "",  # Will be calculated
                    "download_size": 0,  # Will be calculated
                    "download_url": "",
                    "dependencies": [
                        {
                            "plugin": "kipy",
                            "version": ">=0.1.0"
                        }
                    ]
                }
            ]
        }
    
    def copy_core_files(self, package_dir: Path, package_config: Dict):
        """Copy core plugin files"""
        logger.info(f"📂 Copying core files for {package_config['name']}...")
        
        # Source directory first
        src_dir = self.project_root / "src"
        if src_dir.exists():
            package_src = package_dir / "src"
            shutil.copytree(src_dir, package_src)
            logger.info(f"✓ Copied source directory: {len(list(src_dir.glob('*.py')))} files")
        
        # Main plugin file (create entry point)
        main_file = self.project_root / package_config['main_file']
        if main_file.exists():
            # Create a simple entry point that imports from src
            entry_point = f"""#!/usr/bin/env python3
'''
{package_config['name']} - PCB Autorouting Plugin
Entry point for the OrthoRoute application
'''

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import and run the main application
import orthoroute

if __name__ == "__main__":
    sys.exit(orthoroute.main())
"""
            with open(package_dir / "__init__.py", 'w', encoding='utf-8') as f:
                f.write(entry_point)
            logger.info(f"✓ Created entry point from {main_file.name}")
        
        # Assets
        assets_dir = self.project_root / "assets"
        if assets_dir.exists():
            package_assets = package_dir / "assets"
            shutil.copytree(assets_dir, package_assets)
            logger.info(f"✓ Copied assets: {len(list(assets_dir.glob('*')))} files")
        
        # Requirements
        requirements_file = self.project_root / "requirements.txt"
        if requirements_file.exists():
            shutil.copy2(requirements_file, package_dir)
            logger.info("✓ Copied requirements.txt")
    
    def copy_optional_files(self, package_dir: Path, package_config: Dict):
        """Copy optional files based on package configuration"""
        
        # Documentation
        if package_config.get('include_docs', False):
            docs_dir = self.project_root / "docs"
            if docs_dir.exists():
                package_docs = package_dir / "docs"
                shutil.copytree(docs_dir, package_docs)
                logger.info(f"✓ Copied documentation: {len(list(docs_dir.glob('*.md')))} files")
            
            # README
            readme_file = self.project_root / "README.md"
            if readme_file.exists():
                shutil.copy2(readme_file, package_dir)
                logger.info("✓ Copied README.md")
        
        # Tests
        if package_config.get('include_tests', False):
            tests_dir = self.project_root / "tests"
            if tests_dir.exists():
                package_tests = package_dir / "tests"
                shutil.copytree(tests_dir, package_tests)
                logger.info(f"✓ Copied tests: {len(list(tests_dir.glob('*.py')))} files")
        
        # GPU acceleration files
        if package_config.get('include_gpu', True):
            # GPU files are already in src/, just log
            logger.info("✓ GPU acceleration included")
        else:
            # Remove GPU-specific files for lite version
            gpu_file = package_dir / "src" / "gpu_routing_engine.py"
            if gpu_file.exists():
                gpu_file.unlink()
                logger.info("✓ Removed GPU files for lite version")
    
    def create_package_zip(self, package_dir: Path, package_config: Dict) -> Path:
        """Create ZIP package"""
        zip_name = f"{package_config['name']}-{self.version}.zip"
        zip_path = self.build_dir / zip_name
        
        logger.info(f"📦 Creating ZIP package: {zip_name}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    # Create archive path relative to package directory
                    arcname = package_config['name'] + '/' + str(file_path.relative_to(package_dir))
                    zipf.write(file_path, arcname)
        
        # Calculate size
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Package created: {zip_name} ({size_mb:.2f} MB)")
        
        return zip_path
    
    def build_package(self, package_type: str) -> Optional[Path]:
        """Build a specific package type"""
        if package_type not in self.packages:
            logger.error(f"Unknown package type: {package_type}")
            return None
        
        package_config = self.packages[package_type]
        logger.info(f"🏗️ Building {package_type} package: {package_config['name']}")
        
        # Create package directory
        package_dir = self.build_dir / package_config['name']
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy files
        self.copy_core_files(package_dir, package_config)
        self.copy_optional_files(package_dir, package_config)
        
        # Create metadata
        metadata = self.create_plugin_metadata(package_config)
        metadata_file = package_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("✓ Created metadata.json")
        
        # Create ZIP package
        zip_path = self.create_package_zip(package_dir, package_config)
        
        logger.info(f"✅ {package_type.title()} package complete: {zip_path}")
        return zip_path
    
    def build_all_packages(self) -> List[Path]:
        """Build all package types"""
        logger.info("🚀 Starting OrthoRoute build process...")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Version: {self.version}")
        
        self.clean_build_directory()
        
        built_packages = []
        for package_type in self.packages:
            try:
                zip_path = self.build_package(package_type)
                if zip_path:
                    built_packages.append(zip_path)
            except Exception as e:
                logger.error(f"Failed to build {package_type} package: {e}")
        
        # Summary
        logger.info("="*60)
        logger.info("🎉 BUILD COMPLETE!")
        logger.info(f"Built {len(built_packages)} packages:")
        
        total_size = 0
        for package_path in built_packages:
            size_mb = package_path.stat().st_size / (1024 * 1024)
            total_size += size_mb
            logger.info(f"  📦 {package_path.name} ({size_mb:.2f} MB)")
        
        logger.info(f"Total size: {total_size:.2f} MB")
        logger.info(f"Build directory: {self.build_dir}")
        logger.info("="*60)
        
        return built_packages

def main():
    """Main build script entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OrthoRoute Build System')
    parser.add_argument('--package', choices=['production', 'lite', 'development', 'all'], 
                       default='all', help='Package type to build')
    parser.add_argument('--version', default='1.0.0', help='Version number')
    parser.add_argument('--clean', action='store_true', help='Clean build directory only')
    
    args = parser.parse_args()
    
    builder = OrthoRouteBuildSystem()
    builder.version = args.version
    
    if args.clean:
        builder.clean_build_directory()
        return 0
    
    if args.package == 'all':
        builder.build_all_packages()
    else:
        builder.build_package(args.package)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
