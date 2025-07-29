"""
Load the full OrthoRoute plugin
"""

import os
import sys

# Add the plugin directory to Python path
plugin_dir = os.path.dirname(os.path.abspath(__file__))
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)

# Import the full plugin
exec(open(os.path.join(plugin_dir, "full_plugin.py")).read())
