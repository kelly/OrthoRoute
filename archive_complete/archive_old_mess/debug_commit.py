#!/usr/bin/env python3
"""
Debug commit methods
"""

import time

def log_message(message):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

try:
    from kipy import KiCad
    
    kicad = KiCad()
    board = kicad.get_board()
    
    if board:
        # Check commit related methods
        commit_methods = [attr for attr in dir(board) if 'commit' in attr.lower()]
        log_message(f"Commit-related methods: {commit_methods}")
        
        # Try to see the help for begin_commit
        try:
            log_message(f"begin_commit signature: {board.begin_commit.__doc__}")
        except:
            pass
            
        # Try to see the help for push_commit
        try:
            log_message(f"push_commit signature: {board.push_commit.__doc__}")
        except:
            pass
        
        # Try the actual commit workflow
        try:
            commit = board.begin_commit()
            log_message(f"✅ begin_commit() returned: {type(commit)}")
            
            # Try to push with the commit object
            board.push_commit(commit)
            log_message("✅ push_commit(commit) worked")
            
        except Exception as e:
            log_message(f"❌ Commit workflow failed: {e}")
            try:
                board.drop_commit()
            except:
                pass
        
except Exception as e:
    log_message(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
