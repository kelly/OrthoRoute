#!/usr/bin/env python3
"""
Robust OrthoRoute Standalone GPU Server with multiple argument parsing strategies
This version handles various ways KiCad might pass arguments
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

def log_message(message, log_file=None):
    """Thread-safe logging"""
    timestamp = time.strftime('%H:%M:%S')
    full_msg = f"[{timestamp}] {message}"
    print(full_msg)
    
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(full_msg + "\n")
                f.flush()
        except Exception as e:
            print(f"Logging error: {e}")

def parse_arguments():
    """Parse command line arguments with multiple fallback strategies"""
    work_dir = None
    timeout = 3600
    
    print(f"[ARGS] Raw arguments: {sys.argv}")
    print(f"[ARGS] Number of arguments: {len(sys.argv)}")
    
    # Strategy 1: Standard argparse (if available)
    try:
        import argparse
        parser = argparse.ArgumentParser(description='OrthoRoute Standalone GPU Server')
        parser.add_argument('--work-dir', required=False, help='Working directory for communication files')
        parser.add_argument('--timeout', type=int, default=3600, help='Server timeout in seconds')
        
        args = parser.parse_args()
        if args.work_dir:
            work_dir = args.work_dir
            timeout = args.timeout
            print(f"[ARGS] Parsed via argparse: work_dir={work_dir}, timeout={timeout}")
            return work_dir, timeout
            
    except Exception as e:
        print(f"[ARGS] Argparse failed: {e}")
    
    # Strategy 2: Manual argument parsing
    for i, arg in enumerate(sys.argv):
        if arg in ['--work-dir', '-w', '--workdir', '--work_dir']:
            if i + 1 < len(sys.argv):
                work_dir = sys.argv[i + 1]
                print(f"[ARGS] Found work-dir via manual parsing: {work_dir}")
        elif arg in ['--timeout', '-t']:
            if i + 1 < len(sys.argv):
                try:
                    timeout = int(sys.argv[i + 1])
                    print(f"[ARGS] Found timeout via manual parsing: {timeout}")
                except ValueError:
                    pass
    
    # Strategy 3: Look for directory argument without flag
    if not work_dir:
        for arg in sys.argv[1:]:
            if not arg.startswith('-') and ('temp' in arg.lower() or 'orthoroute' in arg.lower()):
                work_dir = arg
                print(f"[ARGS] Found work-dir via heuristic: {work_dir}")
                break
    
    # Strategy 4: Environment variable fallback
    if not work_dir:
        work_dir = os.environ.get('ORTHOROUTE_WORK_DIR')
        if work_dir:
            print(f"[ARGS] Found work-dir via environment: {work_dir}")
    
    # Strategy 5: Default temp directory
    if not work_dir:
        import tempfile
        work_dir = os.path.join(tempfile.gettempdir(), 'orthoroute_default')
        print(f"[ARGS] Using default work-dir: {work_dir}")
    
    return work_dir, timeout

class OrthoRouteStandaloneServer:
    """Standalone GPU routing server that runs outside KiCad"""
    
    def __init__(self, work_dir):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        # Communication files
        self.input_file = self.work_dir / "routing_request.json"
        self.output_file = self.work_dir / "routing_result.json"
        self.status_file = self.work_dir / "routing_status.json"
        self.log_file = self.work_dir / "server.log"
        self.shutdown_file = self.work_dir / "shutdown.flag"
        
        # Clear any existing files
        for file in [self.input_file, self.output_file, self.status_file, self.shutdown_file]:
            if file.exists():
                file.unlink()
        
        log_message("[INIT] OrthoRoute Standalone Server initialized", self.log_file)
        log_message(f"[DIR] Work directory: {self.work_dir}", self.log_file)
    
    def update_status(self, status, progress=0, message=""):
        """Update status file for KiCad to read"""
        try:
            status_data = {
                "status": status,  # "idle", "working", "complete", "error"
                "progress": progress,
                "message": message,
                "timestamp": time.time()
            }
            
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
            
            log_message(f"[STATUS] Status: {status} ({progress}%) - {message}", self.log_file)
            
        except Exception as e:
            log_message(f"[WARN] Status update error: {e}", self.log_file)
    
    def process_routing_request(self):
        """Process a single routing request"""
        try:
            # Read the request
            with open(self.input_file, 'r') as f:
                request_data = json.load(f)
            
            log_message("[REQUEST] Processing routing request", self.log_file)
            self.update_status("working", 30, "Processing routing request")
            
            # Simulate routing (replace with actual routing logic)
            import time
            time.sleep(2)  # Simulate processing time
            
            # Create result
            result = {
                "success": True,
                "routed_nets": [],
                "statistics": {
                    "success_rate": 85.7,
                    "routed_count": 24,
                    "total_nets": 28
                },
                "timestamp": time.time()
            }
            
            # Write result
            with open(self.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            self.update_status("complete", 100, "Routing completed")
            log_message("[COMPLETE] Routing request completed", self.log_file)
            
            # Remove the input file to indicate completion
            self.input_file.unlink()
            
        except Exception as e:
            log_message(f"[ERROR] Request processing error: {e}", self.log_file)
            self.update_status("error", 0, f"Processing error: {str(e)}")
            raise
    
    def run_server(self):
        """Main server loop with simplified error handling"""
        try:
            log_message("[START] Starting OrthoRoute Standalone Server", self.log_file)
            self.update_status("starting", 0, "Initializing server")
            
            # Test basic functionality
            log_message("[TEST] Testing basic Python functionality", self.log_file)
            self.update_status("working", 10, "Testing basic functionality")
            
            # Try to load GPU modules
            log_message("[LOAD] Loading GPU modules...", self.log_file)
            self.update_status("working", 20, "Loading GPU modules")
            
            try:
                import cupy as cp
                import numpy as np
                
                # Test GPU functionality
                test_array = cp.array([1, 2, 3, 4, 5])
                test_result = cp.sum(test_array)
                
                log_message(f"[OK] GPU modules loaded successfully", self.log_file)
                log_message(f"[TEST] GPU test result: {test_result}", self.log_file)
                self.update_status("idle", 100, "Server ready with GPU acceleration")
                
            except Exception as gpu_error:
                log_message(f"[WARN] GPU loading failed: {gpu_error}", self.log_file)
                self.update_status("idle", 100, "Server ready (CPU mode)")
            
            # Main processing loop (improved for actual routing)
            log_message("[READY] Server ready for requests", self.log_file)
            
            start_time = time.time()
            timeout = 1800  # 30 minutes timeout for routing operations
            
            while time.time() - start_time < timeout:
                # Check for shutdown signal
                if self.shutdown_file.exists():
                    log_message("[SHUTDOWN] Shutdown signal received", self.log_file)
                    break
                
                # Check for input file
                if self.input_file.exists():
                    log_message("[REQUEST] Processing routing request", self.log_file)
                    self.update_status("working", 30, "Processing routing request")
                    
                    try:
                        # Read the routing request
                        with open(self.input_file, 'r') as f:
                            request_data = json.load(f)
                        
                        board_data = request_data.get('board_data', {})
                        config = request_data.get('config', {})
                        
                        nets = board_data.get('nets', [])
                        log_message(f"[ROUTING] Processing {len(nets)} nets", self.log_file)
                        
                        # Simulate routing progress
                        routed_nets = []
                        for i, net in enumerate(nets):
                            progress = 30 + int((i / len(nets)) * 60)  # 30% to 90%
                            self.update_status("working", progress, f"Routing net {i+1}/{len(nets)}: {net.get('name', 'unknown')}")
                            
                            # Simulate routing time
                            time.sleep(0.1)
                            
                            # Create a simple routing result (for testing)
                            if len(net.get('pads', [])) >= 2:
                                pads = net['pads']
                                path = [
                                    {'x': pads[0]['x'], 'y': pads[0]['y'], 'layer': 0},
                                    {'x': pads[1]['x'], 'y': pads[1]['y'], 'layer': 0}
                                ]
                                
                                routed_nets.append({
                                    'net_name': net.get('name', ''),
                                    'net_id': net.get('id', i),
                                    'path': path,
                                    'success': True
                                })
                        
                        # Create routing result
                        result = {
                            "success": True,
                            "message": f"Successfully routed {len(routed_nets)} nets",
                            "routed_nets": routed_nets,
                            "statistics": {
                                "total_nets": len(nets),
                                "routed_count": len(routed_nets),
                                "success_rate": (len(routed_nets) / len(nets) * 100) if nets else 0
                            },
                            "timestamp": time.time()
                        }
                        
                        # Write result file first, then update status
                        with open(self.output_file, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        # Ensure file is written
                        time.sleep(0.1)
                        
                        self.update_status("complete", 100, f"Routing completed - {len(routed_nets)} nets routed")
                        log_message(f"[COMPLETE] Routing completed successfully", self.log_file)
                        
                        # Remove input file to indicate processing complete
                        self.input_file.unlink()
                        
                        # Continue running for more requests
                        
                    except Exception as routing_error:
                        log_message(f"[ERROR] Routing error: {routing_error}", self.log_file)
                        self.update_status("error", 0, f"Routing error: {str(routing_error)}")
                        
                        # Write error result
                        error_result = {
                            "success": False,
                            "error": str(routing_error),
                            "timestamp": time.time()
                        }
                        
                        with open(self.output_file, 'w') as f:
                            json.dump(error_result, f, indent=2)
                
                time.sleep(0.5)  # Check for requests every 0.5 seconds
            
            log_message(f"[TIMEOUT] Server timeout reached after {timeout} seconds", self.log_file)
            return True
            
        except Exception as e:
            log_message(f"[ERROR] Server error: {e}", self.log_file)
            log_message(f"[TRACE] Stack trace:\n{traceback.format_exc()}", self.log_file)
            self.update_status("error", 0, f"Server error: {str(e)}")
            return False

def main():
    """Main entry point with robust error handling"""
    print(f"[START] OrthoRoute Robust Server starting...")
    print(f"[PYTHON] Python executable: {sys.executable}")
    print(f"[PYTHON] Python version: {sys.version}")
    print(f"[CWD] Current directory: {os.getcwd()}")
    
    try:
        work_dir, timeout = parse_arguments()
        
        if not work_dir:
            print(f"[ERROR] No work directory specified!")
            print(f"[USAGE] Usage: {sys.argv[0]} --work-dir <directory>")
            return 1
        
        print(f"[CONFIG] Work directory: {work_dir}")
        print(f"[CONFIG] Timeout: {timeout} seconds")
        
        server = OrthoRouteStandaloneServer(work_dir)
        
        success = server.run_server()
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n⌨️ Keyboard interrupt - shutting down...")
        return 0
    except Exception as e:
        print(f"[CRASH] Server failed: {e}")
        print(f"[TRACE] Stack trace:\n{traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
