# Minimal Track Test Plugin

A simple KiCad plugin to test basic track creation.

## What it does

Creates exactly **one track** on your board:
- Start: (10mm, 10mm)
- End: (30mm, 10mm)  
- Width: 0.25mm
- Layer: F.Cu (front copper)

## Installation

1. Install in KiCad via Tools → Plugin and Content Manager
2. Restart KiCad
3. Find "Minimal Track Test" in Tools → External Plugins

## Requirements

- KiCad 9.0+
- kicad-python package

If you get import errors, install with:
```
pip install kicad-python
```

## Usage

1. Open any PCB in KiCad
2. Run the plugin
3. Check for a horizontal track at the top-left area of your board

This is purely for testing the KiCad IPC API track creation.
