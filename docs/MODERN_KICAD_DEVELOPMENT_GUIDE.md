# Modern KiCad Plugin Development Guide

## Overview

This guide provides comprehensive information for developing modern KiCad plugins using the IPC API, based on the latest KiCad development practices and the comprehensive KiCad development guide shared by the user.

## KiCad 9.0+ IPC API Architecture

### The Transition from SWIG to IPC

**KiCad 9.0 represents a fundamental architectural shift** from SWIG bindings to the IPC (Inter-Process Communication) API:

#### SWIG Era (Deprecated)
```python
# Old SWIG approach - DO NOT USE
import pcbnew  # Direct memory access, crashes affect KiCad
board = pcbnew.GetBoard()  # Internal API access
```

#### IPC Era (Modern Standard)
```python
# Modern IPC approach - RECOMMENDED
from kipy import KiCad  # Official Protocol Buffer wrappers
kicad = KiCad()  # Process-isolated connection
board = kicad.board.get_board()  # Protocol Buffer communication
```

### Key Benefits of IPC API

1. **Process Isolation**: Plugin crashes cannot affect KiCad
2. **Stable Interface**: Versioned Protocol Buffer definitions
3. **Future Compatibility**: Supported through KiCad 10.0+
4. **Official Support**: Documented and maintained by KiCad team
5. **Better Testing**: Independent process testing capabilities

## Plugin Structure for IPC API

### Modern Plugin.json Structure
```json
{
  "name": "Your Plugin Name",
  "description": "Plugin description",
  "description_full": "Comprehensive description with features",
  "identifier": "com.yourcompany.pluginname",
  "type": "plugin",
  "api": {
    "version": "1.0"
  },
  "author": {
    "name": "Your Name",
    "contact": {
      "web": "https://yourwebsite.com"
    }
  },
  "maintainer": {
    "name": "Your Name",
    "contact": {
      "web": "https://yourwebsite.com"
    }
  },
  "license": "MIT",
  "resources": {
    "icon": "resources/icon.png"
  },
  "versions": [
    {
      "version": "1.0.0",
      "status": "stable",
      "kicad_version": "9.0.0"
    }
  ]
}
```

### Plugin Entry Point Structure
```python
"""
Modern KiCad IPC Plugin Template
"""
import os
import sys
import json
import logging
from pathlib import Path

# Modern IPC imports - official kicad-python package
try:
    from kipy import KiCad
    from kipy.board import Board
    from kipy.track import Track
    IPC_AVAILABLE = True
except ImportError as e:
    IPC_AVAILABLE = False
    import_error = str(e)

class ModernKiCadPlugin:
    """
    Modern KiCad plugin using IPC API
    """
    
    def __init__(self):
        self.name = "Modern Plugin"
        self.description = "Template for modern KiCad plugins"
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for plugin debugging"""
        log_file = Path.home() / "Documents" / "KiCad" / "logs" / f"{self.name.lower().replace(' ', '_')}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(self.name)
    
    def run(self):
        """Main plugin entry point"""
        if not IPC_AVAILABLE:
            self.logger.error(f"IPC API not available: {import_error}")
            self._show_error_dialog("IPC API not available. Please install kicad-python package.")
            return
            
        try:
            # Connect to KiCad via IPC API
            kicad = KiCad()
            board = kicad.board.get_board()
            
            # Perform plugin operations
            self.process_board(board)
            
            # Apply changes back to KiCad
            kicad.board.refresh()
            
        except Exception as e:
            self.logger.error(f"Plugin execution failed: {e}")
            self._show_error_dialog(f"Plugin failed: {e}")
    
    def process_board(self, board):
        """Process the board - implement your logic here"""
        self.logger.info("Processing board...")
        
        # Example: Add a test track
        track = Track()
        track.set_start(10_000_000, 10_000_000)  # 10mm, 10mm in KiCad units
        track.set_end(30_000_000, 10_000_000)    # 30mm, 10mm in KiCad units
        track.set_width(250_000)                 # 0.25mm width
        track.set_layer("F.Cu")
        
        board.create_items([track])
        self.logger.info("Test track created")
    
    def _show_error_dialog(self, message):
        """Show error dialog to user"""
        # In a real plugin, you'd use proper KiCad dialogs
        print(f"ERROR: {message}")

# Plugin registration for KiCad
def main():
    """Main function called by KiCad"""
    plugin = ModernKiCadPlugin()
    plugin.run()

# For direct execution (testing)
if __name__ == "__main__":
    main()
```

## Development Best Practices

### 1. Environment Setup

#### Install kicad-python Package
```bash
# Windows
"C:\Program Files\KiCad\9.0\bin\python.exe" -m pip install kicad-python

# Linux
python3 -m pip install kicad-python

# macOS
/Applications/KiCad/KiCad.app/Contents/Frameworks/Python.framework/Versions/Current/bin/python3 -m pip install kicad-python
```

#### Development Environment
```python
# Create isolated development environment
python -m venv kicad_dev
source kicad_dev/bin/activate  # Linux/macOS
# kicad_dev\Scripts\activate  # Windows

pip install kicad-python
pip install pytest
pip install mypy
pip install black
pip install flake8
```

### 2. Testing Strategy

#### Unit Testing with IPC API
```python
import pytest
from unittest.mock import Mock, patch
from your_plugin import ModernKiCadPlugin

class TestModernPlugin:
    def test_plugin_initialization(self):
        plugin = ModernKiCadPlugin()
        assert plugin.name == "Modern Plugin"
    
    @patch('your_plugin.KiCad')
    def test_board_processing(self, mock_kicad):
        # Mock IPC API responses
        mock_board = Mock()
        mock_kicad.return_value.board.get_board.return_value = mock_board
        
        plugin = ModernKiCadPlugin()
        plugin.process_board(mock_board)
        
        # Verify IPC API calls
        mock_board.create_items.assert_called_once()
```

#### Integration Testing
```python
# Test with actual KiCad CLI
import subprocess
import tempfile
from pathlib import Path

def test_plugin_with_kicad_cli():
    """Test plugin execution with KiCad CLI"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test PCB
        test_pcb = Path(temp_dir) / "test.kicad_pcb"
        create_test_pcb(test_pcb)
        
        # Run plugin via KiCad CLI
        result = subprocess.run([
            "kicad-cli", "pcb", "plugin", "run",
            "--plugin", "your_plugin.py",
            "--input", str(test_pcb)
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "Plugin executed successfully" in result.stdout
```

### 3. Process Isolation Architecture

#### Plugin Process
```python
class PluginProcess:
    """Main plugin process - communicates with KiCad via IPC"""
    
    def __init__(self):
        self.kicad = KiCad()  # IPC connection
        
    def launch_worker_process(self, task_data):
        """Launch isolated worker for complex operations"""
        import subprocess
        import json
        
        # Serialize task data
        task_file = Path.cwd() / "task_data.json"
        with open(task_file, 'w') as f:
            json.dump(task_data, f)
            
        # Launch worker process
        worker_script = Path(__file__).parent / "worker_process.py"
        process = subprocess.Popen([
            sys.executable, str(worker_script), str(task_file)
        ])
        
        return process
```

#### Worker Process
```python
class WorkerProcess:
    """Isolated worker process - no KiCad dependencies"""
    
    def __init__(self, task_file):
        with open(task_file, 'r') as f:
            self.task_data = json.load(f)
            
    def process_task(self):
        """Process task without KiCad dependencies"""
        # Heavy computation, GPU operations, etc.
        result = self.compute_routing(self.task_data)
        
        # Save results for plugin to read
        result_file = Path.cwd() / "task_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f)
            
    def compute_routing(self, data):
        """Example computation that might crash"""
        try:
            import cupy as cp  # GPU operations
            # Perform GPU computation
            return {"status": "success", "tracks": []}
        except Exception as e:
            return {"status": "error", "message": str(e)}
```

### 4. Error Handling and Logging

#### Comprehensive Error Handling
```python
import logging
import traceback
from pathlib import Path

class RobustPlugin:
    def __init__(self):
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path.home() / "Documents" / "KiCad" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(log_dir / f"{self.__class__.__name__}.log")
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def safe_execute(self, operation, *args, **kwargs):
        """Execute operation with comprehensive error handling"""
        try:
            self.logger.info(f"Starting operation: {operation.__name__}")
            result = operation(*args, **kwargs)
            self.logger.info(f"Operation completed successfully: {operation.__name__}")
            return result
            
        except Exception as e:
            self.logger.error(f"Operation failed: {operation.__name__}")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Report to user
            self._show_error_dialog(f"Operation failed: {str(e)}")
            return None
```

### 5. Packaging and Distribution

#### Modern Package Structure
```
your_plugin/
├── plugin.json              # KiCad Plugin Manager metadata
├── metadata.json            # Package metadata
├── plugins/
│   ├── __init__.py          # Plugin entry point
│   ├── main_plugin.py       # Main plugin logic
│   ├── worker_process.py    # Isolated worker
│   └── utils.py             # Utility functions
├── resources/
│   ├── icon.png             # Plugin icon (64x64)
│   └── toolbar_icon.png     # Toolbar icon (24x24)
├── tests/
│   ├── test_plugin.py       # Unit tests
│   └── test_integration.py  # Integration tests
├── docs/
│   ├── README.md            # User documentation
│   └── api_reference.md     # API documentation
└── requirements.txt         # Python dependencies
```

#### Build Script
```python
#!/usr/bin/env python3
"""
Build script for modern KiCad plugin
"""
import zipfile
import json
from pathlib import Path

def build_package():
    """Build plugin package for distribution"""
    package_dir = Path("your_plugin")
    output_file = Path("your-plugin-kicad-addon.zip")
    
    # Validate package structure
    required_files = [
        "plugin.json",
        "plugins/__init__.py",
        "resources/icon.png"
    ]
    
    for required_file in required_files:
        if not (package_dir / required_file).exists():
            raise FileNotFoundError(f"Required file missing: {required_file}")
    
    # Create package
    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in package_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir.parent)
                zf.write(file_path, arcname)
    
    print(f"Package created: {output_file}")
    print(f"Size: {output_file.stat().st_size / 1024:.1f} KB")

if __name__ == "__main__":
    build_package()
```

### 6. Community Resources and Support

#### Official KiCad Resources
- **IPC API Documentation**: https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/
- **Plugin Development Guide**: https://dev-docs.kicad.org/en/plugins/
- **Protocol Buffer Definitions**: https://gitlab.com/kicad/code/kicad/-/tree/master/common/api/
- **kicad-python Package**: https://pypi.org/project/kicad-python/

#### Community Support
- **KiCad Forum Plugin Section**: https://forum.kicad.info/c/development/plugins/
- **GitLab Issues**: https://gitlab.com/kicad/code/kicad/-/issues
- **Discord Plugin Channel**: KiCad official Discord server
- **Reddit**: r/KiCad community

#### Example Repositories
- **Official Plugin Examples**: https://gitlab.com/kicad/addons/official/
- **Community Plugin Registry**: https://github.com/kicad/plugin-registry
- **IPC API Examples**: https://gitlab.com/kicad/code/kicad/-/tree/master/qa/python_scripts/

### 7. Advanced Development Patterns

#### Async IPC Operations
```python
import asyncio
from kipy import KiCad

class AsyncPlugin:
    def __init__(self):
        self.kicad = KiCad()
        
    async def process_board_async(self):
        """Process board with async operations"""
        board = await self.kicad.board.get_board_async()
        
        # Process multiple operations concurrently
        tasks = [
            self.process_tracks_async(board),
            self.process_vias_async(board),
            self.process_components_async(board)
        ]
        
        results = await asyncio.gather(*tasks)
        return results
        
    async def process_tracks_async(self, board):
        """Process tracks asynchronously"""
        tracks = await board.get_tracks_async()
        # Process tracks
        return len(tracks)
```

#### Plugin Configuration System
```python
import json
from pathlib import Path

class PluginConfig:
    """Modern plugin configuration management"""
    
    def __init__(self, plugin_name):
        self.plugin_name = plugin_name
        self.config_dir = Path.home() / "Documents" / "KiCad" / "plugins" / plugin_name
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        
    def load_config(self):
        """Load plugin configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return self.get_default_config()
        
    def save_config(self, config):
        """Save plugin configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def get_default_config(self):
        """Get default configuration"""
        return {
            "version": "1.0.0",
            "enabled": True,
            "settings": {
                "auto_route": False,
                "via_cost": 1.0,
                "track_width": 0.25
            }
        }
```

## Migration from SWIG to IPC

### API Mapping Guide

#### Board Operations
```python
# SWIG (Old)
import pcbnew
board = pcbnew.GetBoard()
tracks = board.GetTracks()

# IPC (New)
from kipy import KiCad
kicad = KiCad()
board = kicad.board.get_board()
tracks = board.get_tracks()
```

#### Track Creation
```python
# SWIG (Old)
import pcbnew
track = pcbnew.PCB_TRACK(board)
track.SetStart(pcbnew.VECTOR2I(10000000, 10000000))
track.SetEnd(pcbnew.VECTOR2I(30000000, 10000000))
board.Add(track)

# IPC (New)
from kipy.track import Track
track = Track()
track.set_start(10_000_000, 10_000_000)
track.set_end(30_000_000, 10_000_000)
board.create_items([track])
```

#### Component Access
```python
# SWIG (Old)
import pcbnew
modules = board.GetFootprints()
for module in modules:
    ref = module.GetReference()

# IPC (New)
components = board.get_components()
for component in components:
    ref = component.get_reference()
```

### Migration Checklist

- [ ] Replace `import pcbnew` with `from kipy import KiCad`
- [ ] Update board access from `pcbnew.GetBoard()` to `kicad.board.get_board()`
- [ ] Convert coordinate system (KiCad units remain the same)
- [ ] Update object creation from direct constructors to factory methods
- [ ] Replace direct board modification with `create_items()` calls
- [ ] Update error handling for IPC exceptions
- [ ] Test with process isolation in mind
- [ ] Update documentation for IPC API

## Conclusion

Modern KiCad plugin development with the IPC API provides a robust, future-proof foundation for creating sophisticated plugins. The process isolation, stable API, and official support make it the recommended approach for all new KiCad plugins.

**Key Takeaways:**
1. Always use IPC API over SWIG bindings
2. Implement comprehensive error handling and logging
3. Use process isolation for complex operations
4. Follow modern Python development practices
5. Leverage the official kicad-python package
6. Test thoroughly with both unit and integration tests
7. Package properly for the KiCad Plugin Manager

This architecture ensures your plugins will continue to work as KiCad evolves and provides the best user experience with crash protection and professional quality.
