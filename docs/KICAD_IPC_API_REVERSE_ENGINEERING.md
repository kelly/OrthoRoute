# KiCad IPC API Reverse Engineering: What The Fuck Is Going On?

**Date**: August 8, 2025  
**Status**: We're using undocumented KiCad 9.0+ APIs that shouldn't exist but work perfectly  
**Summary**: Holy shit, we're writing Python that calls C++ through an IPC bridge

## The Mind-Bending Reality

### What We Thought We Were Doing
- Writing a Python plugin for KiCad
- Using documented KiCad Python APIs
- Following normal plugin development patterns

### What We're Actually Doing
- **Reverse-engineering undocumented KiCad 9.0+ IPC APIs**
- **Writing Python that communicates to C++ classes through process isolation**
- **Using bleeding-edge connectivity APIs that have no official documentation**
- **Essentially doing advanced API archaeology on a live system**

## The Technical Architecture (As We Understand It)

```
┌─────────────────┐    IPC Bridge    ┌─────────────────┐
│   Python Plugin │ ←────────────→   │  KiCad C++ Core │
│ (Our Code)      │                  │                 │
│                 │    JSON/Protocol │                 │
│ - board object  │ ←────────────→   │ - BOARD         │
│ - GetConnectivity()│               │ - CONNECTIVITY_DATA│
│ - GetRatsnestForNet()│             │ - RN_NET        │
│ - GetEdges()    │                  │ - CN_EDGE       │
└─────────────────┘                  └─────────────────┘
```

### Process Isolation Architecture
KiCad 9.0+ runs plugins in **separate processes** to prevent crashes:

1. **KiCad Main Process** (C++)
   - Runs the main KiCad application
   - Contains all the C++ classes (BOARD, CONNECTIVITY_DATA, etc.)
   - Hosts the IPC server

2. **Plugin Process** (Python)
   - Runs our Python plugin code
   - Connects to KiCad via IPC (Inter-Process Communication)
   - Gets "proxy objects" that represent C++ classes

3. **IPC Bridge**
   - Translates Python method calls to C++ method calls
   - Serializes data between processes (JSON/Protocol Buffers)
   - Handles object lifetime and memory management

## The Documentation Crisis

### What Documentation Exists
- ✅ **KiCad 9.0 C++ Doxygen** (Generated August 8, 2025) - Current and complete
- ✅ **KiCad 6.0 Python API** (From 2023) - Outdated but detailed
- ❌ **KiCad 9.0+ Python/IPC API** - **DOESN'T EXIST**

### What We Found
- The **C++ classes exist** (confirmed in today's Doxygen)
- The **IPC bridge works** (our code runs successfully)
- The **Python bindings work** (method calls succeed)
- The **documentation is missing** (we're flying blind)

## How We Reverse-Engineered The APIs

### 1. Started With KiCad 6.0 Documentation
```python
# From old docs - these methods existed
board.GetConnectivity()
connectivity.GetRatsnestForNet(netcode)
```

### 2. Fixed Method Name Casing
```python
# Old: get_connectivity() (Python style)
# New: GetConnectivity() (C++ style)
```

### 3. Discovered Object Hierarchy
```python
board = get_current_board()  # Returns C++ BOARD proxy
connectivity = board.GetConnectivity()  # Returns C++ CONNECTIVITY_DATA proxy  
rn_net = connectivity.GetRatsnestForNet(netcode)  # Returns C++ RN_NET proxy
edges = rn_net.GetEdges()  # Returns list of C++ CN_EDGE proxies
```

### 4. Found Working Methods Through Trial and Error
```python
# These work but are undocumented:
edge.IsVisible()        # Returns boolean
edge.GetSourcePos()     # Returns coordinate
edge.GetTargetPos()     # Returns coordinate
```

## The C++ Classes We're Actually Using

From today's KiCad 9.0 Doxygen, these C++ classes exist:

### BOARD
- Main board container class
- `GetConnectivity()` - Returns connectivity data

### CONNECTIVITY_DATA  
- Handles electrical connectivity calculations
- `GetRatsnestForNet(netcode)` - Returns ratsnest for specific net

### RN_NET
- Represents ratsnest for a single net  
- `GetEdges()` - Returns list of unconnected edges

### CN_EDGE
- Represents point-to-point connection (airwire)
- `IsVisible()` - Whether edge should be shown
- `GetSourcePos()` - Start coordinate
- `GetTargetPos()` - End coordinate

## The IPC Magic

### How Method Calls Work
1. **Python**: `board.GetConnectivity()`
2. **IPC Bridge**: Serializes call to KiCad process
3. **KiCad C++**: Executes `BOARD::GetConnectivity()`
4. **IPC Bridge**: Creates proxy object for result
5. **Python**: Receives connectivity proxy object

### Data Serialization
- Coordinates get converted between Python tuples and C++ VECTOR2I
- Object references become proxy handles
- Method calls become IPC messages

## Why This Works

### Process Isolation Benefits
- **Plugin crashes don't kill KiCad**
- **Memory isolation** - plugin memory leaks don't affect KiCad
- **Security** - plugins can't directly corrupt KiCad memory
- **Stability** - bad plugins are contained

### API Consistency  
- **C++ methods map directly to Python**
- **Same object hierarchy** maintained through proxies
- **Method signatures preserved** across IPC boundary

## The Performance Reality

### What's Fast
- Method calls on existing objects (cached proxies)
- Small data transfers (coordinates, booleans)
- Bulk operations within single C++ calls

### What's Slow
- Creating new object proxies (IPC overhead)
- Large data transfers (arrays, complex objects)
- Many small calls in loops (IPC latency)

### Our Optimization
We minimize IPC calls by:
```python
# GOOD: Single call that returns bulk data
edges = rn_net.GetEdges()  # One IPC call, returns all edges

# BAD: Many small calls  
for i in range(edge_count):
    edge = rn_net.GetEdge(i)  # Many IPC calls
```

## The Testing Methodology

### How We Validated Unknown APIs
1. **Try the call** - See if method exists
2. **Check return type** - Print what comes back
3. **Test with different inputs** - Edge cases
4. **Compare with KiCad behavior** - Visual verification
5. **Document what works** - Build our own API reference

### Example Discovery Process
```python
# Step 1: Try the method
try:
    result = edge.IsVisible()
    print(f"IsVisible() returned: {result}, type: {type(result)}")
except Exception as e:
    print(f"IsVisible() failed: {e}")

# Step 2: Test with different edges
for edge in edges:
    visible = edge.IsVisible()
    source = edge.GetSourcePos() 
    target = edge.GetTargetPos()
    print(f"Edge: visible={visible}, {source} -> {target}")
```

## Current Status: What We Know Works

### ✅ Confirmed Working APIs
```python
# Board access
board = get_current_board()

# Connectivity system
connectivity = board.GetConnectivity()

# Net ratsnest data  
rn_net = connectivity.GetRatsnestForNet(netcode)
edges = rn_net.GetEdges()

# Edge information
visible = edge.IsVisible()
source_pos = edge.GetSourcePos()  
target_pos = edge.GetTargetPos()
```

### ❓ Probably Exists But Untested
Based on C++ Doxygen, these likely work:
```python
# Layer information
edge.GetSourceLayer()
edge.GetTargetLayer()

# Network information  
rn_net.GetNetCode()
rn_net.GetDisplayName()

# Board information
board.GetNetCount()
board.GetTrackCount()
```

### 🔍 Need To Explore
More C++ classes that probably have Python bindings:
- `CN_CONNECTIVITY_ALGO` - Connectivity algorithms
- `PCB_TRACK` - Track objects
- `PCB_VIA` - Via objects  
- `NETINFO_ITEM` - Net information

## The Implications

### We're Basically KiCad API Archaeologists
- **Documenting undocumented APIs** through reverse engineering
- **Finding working methods** that have no official documentation
- **Building our own API reference** from scratch
- **Using bleeding-edge features** before they're officially released

### This Could Be Incredibly Valuable
- **First movers** on KiCad 9.0+ IPC APIs
- **Deep understanding** of KiCad's internal architecture  
- **Advanced plugin capabilities** beyond what others can do
- **Potential to influence** future KiCad API documentation

## Next Steps

### 1. Systematic API Discovery
Test every method we can find in the C++ Doxygen:
```python
# Test all CONNECTIVITY_DATA methods
connectivity.GetAlgo()
connectivity.IsConnected()
connectivity.GetRatsnest()
```

### 2. Document Everything
Create comprehensive notes on:
- Which methods work
- Parameter types and return values
- Performance characteristics
- Edge cases and limitations

### 3. Build Higher-Level Abstractions
Create wrapper classes that hide IPC complexity:
```python
class OrthoRouteConnectivity:
    def __init__(self, board):
        self.connectivity = board.GetConnectivity()
    
    def get_all_airwires(self):
        """Get all unconnected airwires in optimized way"""
        # Batch operations to minimize IPC calls
```

### 4. Share Knowledge
- Document discoveries for other plugin developers
- Potentially contribute to official KiCad documentation
- Help bridge the gap between C++ and Python ecosystems

## Conclusion: What The Fuck Is Going On?

**We're essentially time travelers** using future KiCad APIs that exist but aren't documented yet. Through process isolation and IPC, we're calling C++ methods from Python in ways that shouldn't be possible but work perfectly.

**This is both exciting and terrifying:**
- ✅ We have access to powerful, undocumented APIs
- ❌ We're using unsupported functionality
- ✅ Our code works better than it should
- ❌ We have no safety net or official support
- ✅ We're pushing the boundaries of what's possible
- ❌ We might break when KiCad updates

**But it's working**, and that's pretty fucking amazing.

---

*"Any sufficiently advanced technology is indistinguishable from magic."*  
*- Arthur C. Clarke*

*"Any sufficiently reverse-engineered API is indistinguishable from documentation."*  
*- Us, apparently*
