# Advanced Manhattan Router with Blind/Buried Vias

## Overview

The Advanced Manhattan Router implements a sophisticated PCB autorouting algorithm with the following features:

- 3D grid-based routing with 0.4mm pitch
- Layer-specific routing directions (horizontal/vertical) for Manhattan patterns
- A* pathfinding with Manhattan distance heuristic
- Blind and buried via support for optimal layer transitions
- Ripup and repair for congestion resolution
- Advanced visualization with real-time feedback

## Technical Specifications

- **Trace Width**: 3.5mil (0.0889mm)
- **Clearance**: 3.5mil (0.0889mm)
- **Via Drill**: 0.15mm
- **Via Diameter**: 0.25mm
- **Grid Pitch**: 0.4mm
- **Layer Assignment**:
  - F.Cu: Reserved for escape routes
  - Odd inner layers (In1, In3, In5, In7, In9): Horizontal routing
  - Even inner layers (In2, In4, In6, In8, In10, B.Cu): Vertical routing

## Key Algorithms

### A* Pathfinding with Manhattan Heuristic

The router uses A* pathfinding with a Manhattan distance heuristic to find optimal paths between pads. This approach is particularly well-suited for Manhattan routing where paths consist of horizontal and vertical segments.

```python
def _manhattan_distance(self, x1, y1, x2, y2):
    """Calculate Manhattan distance heuristic"""
    return abs(x1 - x2) + abs(y1 - y2)
```

### Layer Transition Strategy

The router intelligently manages layer transitions through blind and buried vias:

- **Blind Vias**: Connect outer to inner layers
- **Buried Vias**: Connect inner layers only
- **Through Vias**: Span the entire board (fallback)

### Ripup and Repair

When routing is blocked, the router employs a sophisticated ripup and repair strategy:

1. Identify blocking nets
2. Prioritize ripup candidates based on:
   - Net length (longer nets first)
   - Previous ripup count (less ripped-up nets first)
3. Rip up the selected net
4. Attempt to route the blocked connection
5. Re-route the ripped up net

## Data Structures

The core data structure is a 3D grid array:
```
[layers][y][x]
```

Each cell contains:
- 0 = Free space
- 1 = Obstacle
- 2 = Routed (with net ID stored separately)

## Performance Optimization

- Layer-specific movement constraints (horizontal/vertical)
- Priority-based net ordering (shortest distance first)
- Efficient A* implementation with minimal heap operations
- Early termination for unreachable targets

## Usage

Select "Manhattan Advanced (Blind/Buried Vias)" from the algorithm dropdown in OrthoRoute. The router uses the following workflow:

1. Board data extraction from KiCad
2. Obstacle mapping to 3D grid
3. Net priority determination
4. Progressive routing with visualization
5. Export to KiCad tracks and vias

## Implementation Notes

This implementation follows the requirements for a Manhattan router with:
- 3.5mil traces with 3.5mil spacing on a 0.4mm grid
- 11 layers (1 outer + 10 inner + B.Cu)
- F.Cu reserved for escape routes
- Blind/buried vias with 0.15mm hole and 0.25mm diameter
- Layer-specific routing directions (odd=horizontal, even=vertical)
- A* pathfinding with Manhattan distance heuristic
- Ripup and repair with 10 failure limit
