#!/usr/bin/env python3
"""
Simple Qt Test - Isolate the crash issue
"""
import sys
import os
import logging
from pathlib import Path

# Setup logging
log_file = Path.home() / "Documents" / "qt_test.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_qt_minimal():
    """Test minimal Qt functionality"""
    logger.info("=== Qt Minimal Test ===")
    
    try:
        # Set Qt environment variables for stability
        os.environ.setdefault("QT_OPENGL", "software")
        os.environ.setdefault("QT_QPA_PLATFORM", "windows")
        
        logger.info("Importing PyQt6...")
        from PyQt6.QtCore import Qt, QCoreApplication
        logger.info("PyQt6.QtCore imported")
        
        # Set Qt attributes before creating QApplication
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)
        QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_DisableWindowContextHelpButton, True)
        logger.info("Qt attributes set")
        
        from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton
        logger.info("PyQt6.QtWidgets imported")
        
        # Create Qt application
        logger.info("Creating QApplication...")
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        logger.info("QApplication created")
        
        # Create simple window
        logger.info("Creating simple window...")
        window = QWidget()
        window.setWindowTitle("Qt Test - OrthoRoute Debug")
        window.setMinimumSize(400, 200)
        
        layout = QVBoxLayout(window)
        label = QLabel("Qt is working!")
        button = QPushButton("Close")
        button.clicked.connect(window.close)
        
        layout.addWidget(label)
        layout.addWidget(button)
        
        logger.info("Showing window...")
        window.show()
        
        logger.info("Qt test successful - entering event loop")
        app.exec()
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"PyQt6 not available:\n{e}\n\nInstall with: pip install PyQt6",
            "Qt Test - Import Error",
            0
        )
    except Exception as e:
        logger.error(f"Qt test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        import ctypes
        ctypes.windll.user32.MessageBoxW(
            0,
            f"Qt test failed:\n{e}",
            "Qt Test - Error",
            0
        )

if __name__ == "__main__":
    test_qt_minimal()
