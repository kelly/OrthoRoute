import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import heapq
from collections import deque
import random

class GridPathfinder:
    def __init__(self, width=20, height=20, obstacle_density=0.2):
        self.width = width
        self.height = height
        self.grid = self.create_grid(obstacle_density)
        self.start = (2, 2)
        self.goal = (height-3, width-3)
        self.bound = width + height  # Distance bound for bounded shortest path
        
    def create_grid(self, obstacle_density):
        """Create grid with trace-like obstacles"""
        grid = np.zeros((self.height, self.width))
        
        # Add borders
        grid[0, :] = 1  # Top border
        grid[-1, :] = 1  # Bottom border
        grid[:, 0] = 1  # Left border
        grid[:, -1] = 1  # Right border
        
        # Add horizontal "traces" (obstacles)
        for row in range(2, self.height-2, 3):
            for col in range(2, self.width-3):
                if random.random() < 0.7:  # 70% chance for trace segment
                    grid[row, col] = 1
        
        # Add vertical "traces" 
        for col in range(3, self.width-3, 4):
            for row in range(2, self.height-3):
                if random.random() < 0.6:  # 60% chance for trace segment
                    grid[row, col] = 1
        
        # Ensure start and goal are free
        grid[2, 2] = 0  # Start
        grid[self.height-3, self.width-3] = 0  # Goal
        
        # Clear some paths around start and goal
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if 0 <= 2+dr < self.height and 0 <= 2+dc < self.width:
                    grid[2+dr, 2+dc] = 0
                if 0 <= self.height-3+dr < self.height and 0 <= self.width-3+dc < self.width:
                    grid[self.height-3+dr, self.width-3+dc] = 0
        
        return grid
    
    def get_neighbors(self, pos):
        """Get valid neighboring positions"""
        row, col = pos
        neighbors = []
        
        # 4-directional movement
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.height and 
                0 <= new_col < self.width and 
                self.grid[new_row, new_col] == 0):
                neighbors.append((new_row, new_col))
        
        return neighbors

class DijkstraGridSolver:
    def __init__(self, pathfinder):
        self.pathfinder = pathfinder
        self.finished = False
        self.reset()
    
    def reset(self):
        self.distances = {}
        self.visited = set()
        self.frontier = [(0, self.pathfinder.start)]
        self.parent = {}
        self.current = None
        self.just_explored = []
        self.step_count = 0
        self.finished = False
        
        # Initialize distances
        for i in range(self.pathfinder.height):
            for j in range(self.pathfinder.width):
                self.distances[(i, j)] = float('inf')
        self.distances[self.pathfinder.start] = 0
    
    def step(self):
        """One step of Dijkstra's algorithm"""
        if not self.frontier or self.finished:
            return False
        
        # Extract minimum distance node
        current_dist, current = heapq.heappop(self.frontier)
        
        # Skip if already visited (heap can have duplicates)
        if current in self.visited:
            return len(self.frontier) > 0
        
        self.current = current
        self.visited.add(current)
        self.just_explored = []
        self.step_count += 1
        
        # Check if we found the goal
        if current == self.pathfinder.goal:
            self.finished = True
            return False
        
        # Explore neighbors
        for neighbor in self.pathfinder.get_neighbors(current):
            if neighbor not in self.visited:
                new_dist = current_dist + 1
                
                if new_dist < self.distances[neighbor]:
                    self.distances[neighbor] = new_dist
                    self.parent[neighbor] = current
                    heapq.heappush(self.frontier, (new_dist, neighbor))
                    self.just_explored.append(neighbor)
        
        return True
    
    def get_shortest_path(self):
        """Reconstruct the shortest path from start to goal"""
        if self.pathfinder.goal not in self.parent:
            return []
        
        path = []
        current = self.pathfinder.goal
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        path.reverse()
        return path

class NewAlgorithmGridSolver:
    def __init__(self, pathfinder):
        self.pathfinder = pathfinder
        self.k = max(3, int(np.log(pathfinder.width * pathfinder.height) ** (1/3)))
        self.finished = False
        self.reset()
    
    def reset(self):
        self.distances = {}
        self.visited = set()
        self.working_set = set([self.pathfinder.start])
        self.pivots = set()
        self.current_batch = []
        self.parent = {}
        self.just_explored = []
        self.step_count = 0
        self.phase = "finding_pivots"
        self.pivot_steps = 0
        self.finished = False
        
        # Initialize distances
        for i in range(self.pathfinder.height):
            for j in range(self.pathfinder.width):
                self.distances[(i, j)] = float('inf')
        self.distances[self.pathfinder.start] = 0
    
    def step(self):
        """One step of the new algorithm"""
        if self.finished:
            return False
            
        if self.phase == "finding_pivots":
            return self._pivot_finding_step()
        elif self.phase == "processing_batch":
            return self._batch_processing_step()
        else:
            return False
    
    def _pivot_finding_step(self):
        """Find pivots using k-step exploration"""
        if self.pivot_steps >= self.k:
            self._identify_pivots()
            self.phase = "processing_batch"
            return True
        
        # One step of parallel exploration from working set
        new_nodes = set()
        self.just_explored = []
        
        current_working = list(self.working_set - self.visited)
        
        if not current_working:
            self._identify_pivots()
            self.phase = "processing_batch"
            return True
        
        # Take first few nodes from working set and explore them
        nodes_to_explore = current_working[:2]  # Explore 2 nodes at once
        
        for node in nodes_to_explore:
            neighbors = self.pathfinder.get_neighbors(node)
            for neighbor in neighbors:
                if neighbor not in self.visited:
                    new_dist = self.distances[node] + 1
                    
                    if new_dist < self.distances[neighbor]:
                        self.distances[neighbor] = new_dist
                        self.parent[neighbor] = node
                        new_nodes.add(neighbor)
                        self.just_explored.append(neighbor)
        
        self.working_set.update(new_nodes)
        self.pivot_steps += 1
        return True
    
    def _identify_pivots(self):
        """Identify key nodes as pivots"""
        # Select nodes from working set that aren't visited yet
        candidates = [node for node in self.working_set 
                     if node not in self.visited and self.distances[node] < float('inf')]
        
        if not candidates:
            return
        
        # Sort by distance from start, but EXCLUDE the start node itself
        candidates = [node for node in candidates if node != self.pathfinder.start]
        candidates.sort(key=lambda node: self.distances[node])
        
        pivot_count = min(self.k, len(candidates))
        self.pivots = set(candidates[:pivot_count])
        
        # If no pivots selected, add some nodes from the frontier
        if not self.pivots and self.working_set:
            frontier_nodes = [node for node in self.working_set 
                            if node not in self.visited and node != self.pathfinder.start]
            if frontier_nodes:
                self.pivots.add(frontier_nodes[0])
    
    def _batch_processing_step(self):
        """Process batch of pivots"""
        if not self.pivots and not self.current_batch:
            # If we're out of pivots, add more from the working set
            unvisited_working = [node for node in self.working_set if node not in self.visited]
            if unvisited_working:
                # Add some unvisited nodes as new pivots
                new_pivots = unvisited_working[:self.k]
                self.pivots.update(new_pivots)
            else:
                self.finished = True
                return False
        
        # If we have pivots but no current batch, start a new batch
        if self.pivots and not self.current_batch:
            # Prioritize goal if it's in pivots
            if self.pathfinder.goal in self.pivots:
                self.current_batch = [self.pathfinder.goal]
                self.pivots.remove(self.pathfinder.goal)
            else:
                batch_size = min(2, len(self.pivots))
                self.current_batch = list(self.pivots)[:batch_size]
                for node in self.current_batch:
                    self.pivots.remove(node)
        
        # Process current batch
        if self.current_batch:
            current = self.current_batch.pop(0)
            self.visited.add(current)
            self.step_count += 1
            self.just_explored = []
            
            # Check if we found the goal
            if current == self.pathfinder.goal:
                self.finished = True
                return False
            
            # Explore neighbors
            for neighbor in self.pathfinder.get_neighbors(current):
                if neighbor not in self.visited:
                    new_dist = self.distances[current] + 1
                    
                    if new_dist < self.distances[neighbor]:
                        self.distances[neighbor] = new_dist
                        self.parent[neighbor] = current
                        self.just_explored.append(neighbor)
                        
                        # Add new neighbors to working set AND as potential pivots
                        self.working_set.add(neighbor)
                        if neighbor not in self.pivots and neighbor not in self.visited:
                            self.pivots.add(neighbor)
        
        return True
    
    def get_shortest_path(self):
        """Reconstruct the shortest path from start to goal"""
        if self.pathfinder.goal not in self.parent:
            return []
        
        path = []
        current = self.pathfinder.goal
        while current is not None:
            path.append(current)
            current = self.parent.get(current)
        path.reverse()
        return path

def create_grid_animation():
    """Create animated grid pathfinding comparison"""
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    # Create pathfinder and solvers with trace-like obstacles
    pathfinder = GridPathfinder(width=18, height=12, obstacle_density=0.3)
    dijkstra = DijkstraGridSolver(pathfinder)
    new_algo = NewAlgorithmGridSolver(pathfinder)
    
    # Setup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    def get_cell_color(pos, solver, algorithm_type):
        """Get color for a grid cell based on algorithm state"""
        if pathfinder.grid[pos] == 1:  # Obstacle
            return 'black'
        elif pos == pathfinder.start:
            return 'green'
        elif pos == pathfinder.goal:
            return 'red'
        elif solver.finished and pos in solver.get_shortest_path():
            return 'limegreen'  # Path in bright green
        elif pos in solver.visited:
            return 'darkred'
        elif algorithm_type == "dijkstra":
            if any(pos == node for _, node in solver.frontier):
                return 'orange'
            elif pos == solver.current:
                return 'yellow'
            elif pos in solver.just_explored:
                return 'lightcoral'
            else:
                return 'white'
        else:  # new algorithm
            if pos in solver.current_batch:
                return 'cyan'
            elif pos in solver.pivots:
                return 'purple'
            elif pos in solver.working_set:
                return 'lightblue'
            elif pos in solver.just_explored:
                return 'lightcoral'
            else:
                return 'white'
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        # Step algorithms
        if frame > 0:
            if not dijkstra.finished:
                dijkstra.step()
            
            if not new_algo.finished:
                new_algo.step()
        
        # Create color grids
        dijkstra_colors = np.zeros((pathfinder.height, pathfinder.width, 3))
        new_algo_colors = np.zeros((pathfinder.height, pathfinder.width, 3))
        
        color_map = {
            'black': [0, 0, 0],
            'white': [1, 1, 1],
            'green': [0, 1, 0],
            'red': [1, 0, 0],
            'darkred': [0.25, 0.25, 0.25],
            'orange': [1, 0.5, 0],
            'yellow': [1, 1, 0],
            'lightcoral': [1, 0.8, 0.8],
            'cyan': [0, 1, 1],
            'purple': [0.5, 0, 0.5],
            'lightblue': [0.7, 0.9, 1],
            'limegreen': [0.2, 1, 0.2]  # Bright green for path
        }
        
        for i in range(pathfinder.height):
            for j in range(pathfinder.width):
                pos = (i, j)
                
                # Dijkstra colors
                dijkstra_color = get_cell_color(pos, dijkstra, "dijkstra")
                dijkstra_colors[i, j] = color_map[dijkstra_color]
                
                # New algorithm colors
                new_algo_color = get_cell_color(pos, new_algo, "new")
                new_algo_colors[i, j] = color_map[new_algo_color]
        
        # Display grids
        ax1.imshow(dijkstra_colors, origin='upper')
        ax2.imshow(new_algo_colors, origin='upper')
        
        # Display grids with status info
        dijkstra_status = "FOUND GOAL!" if dijkstra.finished else f"Frontier size: {len(dijkstra.frontier)}"
        ax1.set_title(f"Dijkstra's Algorithm\nStep: {dijkstra.step_count}\n{dijkstra_status}", 
                     fontsize=12, fontweight='bold')
        
        phase_text = new_algo.phase.replace('_', ' ').title()
        new_algo_status = "FOUND GOAL!" if new_algo.finished else f"Pivots: {len(new_algo.pivots)}"
        ax2.set_title(f"New Algorithm O(m log^(2/3) n)\nStep: {new_algo.step_count}\n{new_algo_status}", 
                     fontsize=12, fontweight='bold')
        
        # Add grid lines and clean up axes
        for ax in [ax1, ax2]:
            ax.set_xticks(np.arange(-0.5, pathfinder.width, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, pathfinder.height, 1), minor=True)
            ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            ax.set_xlim(-0.5, pathfinder.width-0.5)
            ax.set_ylim(-0.5, pathfinder.height-0.5)
            # Remove axis numbers
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Add legend
        if frame == 0:
            legend_elements = [
                plt.Rectangle((0,0),1,1, facecolor='green', label='Start'),
                plt.Rectangle((0,0),1,1, facecolor='red', label='Goal'),
                plt.Rectangle((0,0),1,1, facecolor='limegreen', label='Shortest Path'),
                plt.Rectangle((0,0),1,1, facecolor='black', label='Obstacle'),
                plt.Rectangle((0,0),1,1, facecolor='darkred', label='Visited'),
                plt.Rectangle((0,0),1,1, facecolor='orange', label='Frontier (Dijkstra)'),
                plt.Rectangle((0,0),1,1, facecolor='yellow', label='Current (Dijkstra)'),
                plt.Rectangle((0,0),1,1, facecolor='purple', label='Pivots (New)'),
                plt.Rectangle((0,0),1,1, facecolor='cyan', label='Current Batch (New)'),
                plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Working Set (New)'),
            ]
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=5)
        
        fig.suptitle(f"Bounded Shortest Path Comparison - Frame {frame}", fontsize=16, fontweight='bold')
    
    # Create animation with slower speed to see each step
    frames = 150
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)
    
    plt.subplots_adjust(bottom=0.15)
    return ani, pathfinder

if __name__ == "__main__":
    print("Creating grid pathfinding animation...")
    ani, pathfinder = create_grid_animation()
    
    # Save animation
    print("Saving animation...")
    ani.save('grid_pathfinding_comparison.gif', writer='pillow', fps=3)
    
    print("Displaying animation...")
    plt.show()
    
    print("\nAnimation saved as: grid_pathfinding_comparison.gif")
    print("\nWhat you'll see:")
    print("- GREEN: Start point")
    print("- RED: Goal point") 
    print("- LIME GREEN: Shortest path (when found)")
    print("- BLACK: Obstacles (trace-like patterns)")
    print("- LEFT: Dijkstra processing one node at a time, frontier grows large")
    print("- RIGHT: New algorithm finding pivots first, then processing in batches")
    print("- ORANGE: Dijkstra's frontier")
    print("- PURPLE: New algorithm's pivots (reduced frontier)")
    print("- CYAN: Current batch being processed in parallel")
    print("- LIGHTBLUE: Working set during pivot finding")
    print("- DARK RED: Visited nodes")
