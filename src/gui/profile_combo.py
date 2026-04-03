from PySide6.QtWidgets import QComboBox, QMenu, QInputDialog, QMessageBox
from PySide6.QtCore import Signal, Qt

class ProfileComboBox(QComboBox):
    """
    A custom QComboBox for profile selection that supports right-click 
    context menus to New, Save, Duplicate, and Delete profiles natively.
    """
    # Emits (action_type, profile_name) where action_type in ('new', 'save', 'duplicate', 'delete')
    profile_action = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(120)
        
    def contextMenuEvent(self, event):
        if self.count() == 0:
            return
            
        menu = QMenu(self)
        current = self.currentText()
        # Clean current in case it has dirty '*'
        clean_name = current.replace("*", "").strip()
        
        new_act = menu.addAction("New Profile...")
        save_act = menu.addAction(f"Save '{clean_name}'")
        dup_act = menu.addAction(f"Duplicate '{clean_name}'...")
        menu.addSeparator()
        del_act = menu.addAction(f"Delete '{clean_name}'")
        
        # We don't want to delete the default or if we only have 1
        if self.count() <= 1:
            del_act.setEnabled(False)

        action = menu.exec(event.globalPos())
        if not action:
            return
            
        if action == save_act:
            self.profile_action.emit('save', clean_name)
        elif action == new_act:
            new_name, ok = QInputDialog.getText(self, "New Profile", "Enter profile name:")
            if ok and new_name:
                self.profile_action.emit('new', new_name.strip())
        elif action == dup_act:
            new_name, ok = QInputDialog.getText(self, "Duplicate Profile", "New profile name:", text=f"{clean_name} Copy")
            if ok and new_name:
                self.profile_action.emit('duplicate', new_name.strip())
        elif action == del_act:
            reply = QMessageBox.question(self, "Confirm Delete", f"Are you sure you want to delete profile '{clean_name}'?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                self.profile_action.emit('delete', clean_name)
