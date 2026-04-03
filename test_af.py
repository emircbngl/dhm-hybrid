import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
root_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(root_dir / "src"))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from core.autofocus import FocusMetric

def main():
    app = QApplication(sys.argv)
    
    # Initialize main window (won't show it)
    win = MainWindow()
    
    # Mock an image load (512x512 with some features)
    dummy_img = np.zeros((256, 256), dtype=np.uint8)
    dummy_img[100:150, 100:150] = 255
    win._loaded_array = dummy_img
    
    print("Testing regular autofocus...")
    # Enable autofocus mock state
    win._autofocus()
    
    print("Status bar:", win.status_bar.currentMessage())
    print("Main Autfocus Table produced:", hasattr(win, '_autofocus_table'))
    
    print("Testing adaptive AF...")
    # Trigger adaptive focus
    win._trigger_adaptive_focus()
    print("Adaptive AF worker started:", win._adaptive_worker.isRunning())
    
    # We must explicitly wait for worker thread to finish
    win._adaptive_worker.wait()
    print("Adaptive worker finished.")
    
    print("All GUI tests complete.")
    app.quit()

if __name__ == '__main__':
    main()
    print("OK!")
