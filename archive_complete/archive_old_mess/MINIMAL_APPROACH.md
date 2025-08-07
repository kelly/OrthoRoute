# Minimal IPC Plugin Approach - August 4, 2025

## Strategy: Start Simple, Build Up

Instead of debugging the complex GPU routing system, we're taking a minimal approach:

## ✅ What We Built

### 1. Minimal Test Plugin (`minimal-track-test.zip` - 2.3 KB)
- **Purpose**: Test basic KiCad IPC API functionality
- **What it does**: Draws exactly one track from (10mm,10mm) to (30mm,10mm)
- **Dependencies**: Only `kicad-python` package
- **Code**: 50 lines of pure IPC API calls

### 2. Simple Test Process
1. Install `kicad-python` package
2. Install minimal plugin 
3. Test with one track creation
4. If it works → IPC API is functional
5. If it fails → Fix IPC setup before trying complex system

## 🎯 Benefits of This Approach

### Isolates the Problem
- **Complex System**: GPU routing + IPC API + Process isolation + JSON communication
- **Minimal System**: Just IPC API + one track creation
- **Result**: If minimal fails, we know IPC API is the issue

### Easy to Debug
- **50 lines of code** vs **thousands**
- **Clear error messages** from simple operations
- **No GPU dependencies** to complicate things
- **No server processes** to manage

### Validates Core Functionality
- Tests that `kipy` imports work
- Tests that KiCad connection works
- Tests that `Track` creation works
- Tests that `board.create_items()` works
- Tests that board saving works

## 📦 Package Structure

```
minimal-track-test.zip (2.3 KB)
├── plugin.json          # IPC plugin definition
├── metadata.json        # Package metadata  
├── minimal_track_test.py # Main plugin (50 lines)
└── README.md            # Usage instructions
```

## 🧪 The Minimal Plugin Code

```python
#!/usr/bin/env python3
def main():
    from kipy import KiCad
    from kipy.board_types import Track
    from kipy.util.units import from_mm
    from kipy.geometry import Vector2
    
    kicad = KiCad()
    board = kicad.get_board()
    
    track = Track()
    track.start = Vector2(from_mm(10), from_mm(10))
    track.end = Vector2(from_mm(30), from_mm(10))
    track.width = from_mm(0.25)
    track.layer = 0
    
    board.create_items([track])
    board.push_commit("Test track")
    board.save()
    
    print("🎉 SUCCESS!")

if __name__ == "__main__":
    main()
```

## 🚀 Next Steps

1. **User installs** `minimal-track-test.zip`
2. **Tests basic functionality** with one track
3. **If it works** → Proceed to full OrthoRoute system
4. **If it fails** → Debug IPC API setup first

This approach eliminates 99% of the complexity while testing the core functionality that everything else depends on.

## 🎉 Success Criteria

**Minimal plugin works** = KiCad IPC API is properly configured
**Minimal plugin fails** = Fix IPC API before attempting complex routing

---

**This is exactly what you asked for**: "alright you should do all of this with the minimal plugin that just draws one trace" ✅
