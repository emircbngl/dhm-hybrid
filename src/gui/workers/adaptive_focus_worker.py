from PySide6.QtCore import QThread, Signal
import numpy as np

from core.autofocus import AdaptiveFocusState
from core.reconstruction import ReconstructionParams, ReconstructionMethod

class AdaptiveFocusWorker(QThread):
    """
    Background worker that runs a short-range autofocus sweep to track drift,
    emitting the newly discovered optimal Z focus.
    """
    new_z_discovered = Signal(float)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: AdaptiveFocusState = None
        self.field: np.ndarray = None
        self.base_params: ReconstructionParams = None
        self.method: ReconstructionMethod = None
        self._is_running = False
        
    def setup(self, state, field, base_params, method):
        self.state = state
        self.field = field
        self.base_params = base_params
        self.method = method

    def run(self):
        if self.state is None or self.field is None:
            self.error_occurred.emit("Worker lacking state/field parameters.")
            return
            
        self._is_running = True
        try:
            # Perform the mathematical tracking sweep
            best_z = self.state.step(self.field, self.base_params, self.method)
            self.new_z_discovered.emit(best_z)
        except Exception as e:
            self.error_occurred.emit(f"Adaptive focus failed: {str(e)}")
        finally:
            self._is_running = False
