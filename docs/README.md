# OrthoRoute Documentation

This directory contains comprehensive documentation for the OrthoRoute PCB autorouting plugin.

## 📖 Documentation Index

### 🚀 Getting Started
- **[Installation Guide](INSTALL.md)** - Complete setup and installation instructions
- **[Installation (Legacy)](installation.md)** - Alternative installation methods

### 🏗️ Architecture & Design
- **[Modular Architecture](MODULAR_ARCHITECTURE.md)** - Complete system design and component overview
- **[API Reference](api_reference.md)** - Detailed API documentation

### 🔧 Advanced Topics
- **[DRC Extraction](DRC_EXTRACTION.md)** - Design rule checking implementation
- **[KiCad IPC API](KICAD_IPC_API_REVERSE_ENGINEERING.md)** - Deep dive into KiCad integration
- **[IPC API Transition](ipc_api_transition.md)** - Migration to modern KiCad APIs
- **[Advanced IPC Usage](ADVANCED_IPC_API_USAGE.md)** - Advanced integration patterns

### 🧮 Algorithms & Implementation
- **[Frontier Reduction Algorithm](FRONTIER_REDUCTION_ALGORITHM.md)** - Advanced routing algorithms
- **[Frontier Reduction Q&A](FRONTIER_REDUCTION_QA.md)** - Technical deep dive
- **[Board Filename Implementation](BOARD_FILENAME_IMPLEMENTATION.md)** - File handling details

### 💻 Development
- **[Modern KiCad Development Guide](MODERN_KICAD_DEVELOPMENT_GUIDE.md)** - Best practices for KiCad plugins
- **[Contributing Guidelines](contributing.md)** - How to contribute to the project
- **[Practical Applications](PRACTICAL_APPLICATIONS.md)** - Real-world usage examples

### 🔬 Development Tools
- **[Algorithm Visualization](algoviz.py)** - Visual debugging tools for routing algorithms

## 📚 Quick Reference

### Core Components
```
src/
├── core/                      # Core infrastructure
│   ├── drc_rules.py          # DRC rules management
│   ├── gpu_manager.py        # GPU acceleration
│   └── board_interface.py    # Board data abstraction
├── routing_engines/           # Pluggable routing algorithms
│   ├── base_router.py        # Abstract router interface
│   └── lees_router.py        # Lee's wavefront implementation
└── autorouter_factory.py     # Main factory interface
```

### Key Features
- **GPU Acceleration**: 6.7x performance improvement
- **Production DRC**: Proper edge-based clearance calculations
- **Modular Architecture**: Clean separation of routing algorithms
- **KiCad Integration**: Full IPC API support

### Usage Example
```python
from autorouter_factory import create_autorouter, RoutingAlgorithm

# Create autorouter
autorouter = create_autorouter(
    board_data=board_data,
    kicad_interface=kicad_interface,
    algorithm=RoutingAlgorithm.LEE_WAVEFRONT
)

# Route all nets
stats = autorouter.route_all_nets(timeout_per_net=5.0)
```

## 🤝 Contributing

Found an issue with the documentation? Want to add more details? Please see the [Contributing Guidelines](contributing.md) for how to help improve OrthoRoute.

## 📞 Support

- **Issues**: Report bugs on [GitHub Issues](https://github.com/bbenchoff/OrthoRoute/issues)
- **Discussions**: Join the conversation on [GitHub Discussions](https://github.com/bbenchoff/OrthoRoute/discussions)
- **Documentation**: Improve these docs via pull requests
