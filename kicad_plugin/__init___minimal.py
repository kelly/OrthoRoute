"""
Simple test to load minimal plugin
"""

import os
import sys

# Add the plugin directory to Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# Import the minimal plugin
exec(open(os.path.join(plugin_dir, "minimal_plugin_only.py")).read())
