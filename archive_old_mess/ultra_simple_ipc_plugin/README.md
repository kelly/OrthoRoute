# Ultra Simple IPC Test Plugin

A minimal test plugin that demonstrates KiCad 9.0+ IPC plugin system with toolbar button integration.

## Features

- Pure IPC API (no SWIG dependencies)
- Toolbar button integration
- Comprehensive logging for debugging
- KiCad 9.0+ compatible

## Installation

1. Install via KiCad's Plugin and Content Manager (PCM)
2. Or manually copy to your KiCad plugins directory

## Usage

1. Look for "Ultra Simple Test" button in KiCad toolbar
2. Click to execute the plugin
3. Check the log file at `~/kicad_plugin_test.log` for debugging info

## Requirements

- KiCad 9.0+
- Python 3.8+
- kicad-python package (installed automatically)
