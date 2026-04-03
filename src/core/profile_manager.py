import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from PySide6.QtCore import QObject, Signal

class ProfileManager(QObject):
    """
    Manages loading, saving, and enumerating JSON profile settings for Phase 8.
    Split into two distinct categories: 'setup' and 'camera'.
    Emits `profiles_changed(target_type)` when disk structures modify so UIs update synchronously.
    """
    profiles_changed = Signal(str) 
    
    def __init__(self):
        super().__init__()
        self.base_dir = Path.home() / ".dhm-reconstruction" / "profiles"
        self.setup_dir = self.base_dir / "setup"
        self.camera_dir = self.base_dir / "camera"
        
        self._ensure_directories()
        
    def _ensure_directories(self):
        self.setup_dir.mkdir(parents=True, exist_ok=True)
        self.camera_dir.mkdir(parents=True, exist_ok=True)
        
    def list_profiles(self, profile_type: str) -> List[str]:
        target_dir = self.setup_dir if profile_type == "setup" else self.camera_dir
        return [p.stem for p in target_dir.glob("*.json")]
        
    def load_profile(self, profile_type: str, name: str) -> Dict[str, Any]:
        target_dir = self.setup_dir if profile_type == "setup" else self.camera_dir
        path = target_dir / f"{name}.json"
        if not path.exists():
            return {}
            
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load {profile_type} profile '{name}': {e}")
            return {}
            
    def save_profile(self, profile_type: str, name: str, data: Dict[str, Any]):
        target_dir = self.setup_dir if profile_type == "setup" else self.camera_dir
        path = target_dir / f"{name}.json"
        
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self.profiles_changed.emit(profile_type)
        except Exception as e:
            logging.error(f"Failed to save {profile_type} profile '{name}': {e}")
            
    def delete_profile(self, profile_type: str, name: str):
        target_dir = self.setup_dir if profile_type == "setup" else self.camera_dir
        path = target_dir / f"{name}.json"
        
        if path.exists():
            try:
                path.unlink()
                self.profiles_changed.emit(profile_type)
            except Exception as e:
                logging.error(f"Failed to delete {profile_type} profile '{name}': {e}")
