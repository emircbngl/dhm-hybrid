import sys
import os
import numpy as np

# Add Hybrid/src directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow
from PySide6.QtCore import QTimer

def test():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # Simulate loading an image
    win._loaded_array = np.random.rand(512, 512).astype(np.float32)
    from pathlib import Path
    win._loaded_path = Path("dummy.tif")
    win._frame_counter = 1

    def on_worker_error(msg):
        print(f"!!! WORKER ERROR !!! : {msg}")
        app.quit()
        
    def on_recon_completed(res):
        print("Recon completed successfully.")
        app.quit()
    
    win._recon_worker.error_occurred.connect(on_worker_error)
    win._recon_worker.recon_completed.connect(on_recon_completed)

    def trigger_tests():
        print("Starting reconstruction...")
        win._trigger_reconstruction()
        
    QTimer.singleShot(500, trigger_tests)
    app.exec()

if __name__ == "__main__":
    test()
