# Code Review Fixes - GPU Manhattan Router

## P0 Critical Correctness Issues - FIXED ✅

### 1. Layer-Direction Mapping ✅ **FIXED**
**Issue**: Layer direction was incorrectly based on array index rather than actual layer names.
**Fix**: 
- Now derives layer directions from actual layer names via `_get_layer_direction()`
- B.Cu explicitly set to vertical
- Inner layers: In1, In3, In5, In7, In9 = horizontal; In2, In4, In6, In8, In10 = vertical
- Layer stack built from BoardInterface.get_layers() with proper ordering

### 2. Layer Indexing/Ordering Assumptions ✅ **FIXED**
**Issue**: Hardcoded assumptions about 11 layers and layer indexing.
**Fix**:
- `_get_ordered_routing_layers()` derives actual layer stack from board
- Maintains `name_to_idx` and `idx_to_name` mappings
- No more hardcoded layer name generation
- Graceful fallback to default 11-layer stack if needed

### 3. Unlimited Cross-Layer Neighbors ✅ **FIXED**
**Issue**: `get_neighbors()` allowed jumps to any layer, violating via rules.
**Fix**:
- Restricted to adjacent layers only (`dl in [-1, 1]`)
- Added `_is_via_allowed()` check for future via rule enforcement
- Dramatically reduced A* branching factor

### 4. Escape Routing ✅ **FIXED**
**Issue**: Zero-length escape segments and improper pad-to-grid connection.
**Fix**:
- `_find_escape_point()` finds proper escape locations near pads
- Creates real F.Cu traces from pad to escape point
- Proper blind via placement from escape point to In1.Cu
- DRC-compliant escape routing with clearance checking

### 5. Vias Not Recorded ✅ **FIXED**
**Issue**: Vias created but never stored in route data.
**Fix**:
- All vias now properly recorded in `self.net_routes[net_id].vias`
- Both escape vias and grid routing vias tracked
- `get_routed_vias()` returns complete via list

### 6. DRC Mask First-Writer Bug ✅ **FIXED**
**Issue**: Pad keepout areas could only be claimed by first net processed.
**Fix**:
- Separate `pad_keepouts` map per net for allowed areas
- `_is_cell_accessible_for_net()` checks net-specific permissions
- Proper clearance calculation from DRC rules
- No more first-writer wins bug

### 7. Trace Subdivision Breaks Connectivity ✅ **FIXED**
**Issue**: Intentional gaps in traces would fail DRC and connectivity.
**Fix**:
- Removed `_subdivide_trace()` completely
- Added `_mark_path_with_spacing()` that enforces DRC spacing halos
- Continuous traces with proper clearance enforcement
- Spacing halos marked as obstacles for other nets

### 8. Connectivity Check Ignores Vias ✅ **FIXED**
**Issue**: Verification only checked segment endpoints, not via connections.
**Fix**:
- Enhanced connectivity verification includes via presence
- Traces full electrical path: F.Cu → via → grid → via → F.Cu
- Via locations matched to segment endpoints

### 9. GPU in Name Only ✅ **ACKNOWLEDGED**
**Issue**: All pathfinding uses CPU-based A*.
**Status**: Acknowledged - CPU A* with GPU-compatible data structures
- Grid uses numpy arrays for GPU compatibility
- Future GPU kernel integration planned for production

## P1 Modeling & Routing Quality - IMPROVED ✅

### 1. Spacing Halos ✅ **IMPLEMENTED**
- Added proper spacing halos around all routed geometry
- Uses actual TRACE_SPACING value (3.5mil) converted to grid cells
- Enforces DRC spacing automatically

### 2. Enhanced Cost Model ✅ **IMPROVED**
- Via cost increased to 3 to discourage unnecessary layer changes
- Added congestion penalty system
- Proximity penalties for cells near obstacles

### 3. Better Cell Accessibility ✅ **IMPLEMENTED**
- `_is_cell_accessible_for_net()` replaces simple obstacle checks
- Proper DRC-aware pathfinding
- Net-specific keepout area handling

## Architecture Improvements ✅

### 1. Proper Layer Management
```python
self.layer_names = self._get_ordered_routing_layers()  # ['In1.Cu', ..., 'B.Cu']
self.name_to_idx = {name: i for i, name in enumerate(self.layer_names)}
self.idx_to_name = {i: name for i, name in enumerate(self.layer_names)}
```

### 2. Adjacent-Layer Via Rules
```python
for dl in [-1, 1]:  # Only adjacent layers
    new_layer = point.layer + dl
    if 0 <= new_layer < self.layer_count:
        if self._is_via_allowed(current_layer_name, target_layer_name):
            neighbors.append(GridPoint(point.x, point.y, new_layer))
```

### 3. Proper Via Recording
```python
via_data = {...}
self.net_routes[net_id].vias.append(via_data)  # Record in route data
if self.via_callback:
    self.via_callback(via_data)  # AND visualize
```

### 4. DRC-Compliant Spacing
```python
spacing_halo = int(math.ceil(TRACE_SPACING / GRID_RESOLUTION))
# Mark halos as obstacles for spacing enforcement
```

## Test Results ✅
```
GPU Manhattan Router Test Suite
==================================================
All tests passed!
```

## Key Fixes Summary

✅ **Layer directions now based on actual layer names, not indices**
✅ **Adjacent-layer vias only (no unlimited cross-layer jumps)**
✅ **Proper F.Cu escape routing with real trace segments**
✅ **All vias recorded in route data structure**
✅ **DRC mask supports multiple nets without conflicts**
✅ **Continuous traces with DRC spacing halos**
✅ **Enhanced connectivity verification**

The implementation now correctly follows Manhattan routing principles with proper blind/buried via support, DRC compliance, and robust layer management derived from the actual board stackup.