import logging
import os
from time import time
from typing import Any, Dict

from configs import config, resource_path
from dungeon_despair.domain.encounter import Encounter
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.utils import (
    ActionType,
    Direction,
    EntityEnum,
    get_enum_by_value,
    ModifierType,
)
from dungeon_despair.functions import DungeonCrawlerFunctions
from gptfunctionutil import LibCommand

from PyQt6.QtCore import pyqtSlot, QEvent, QObject, QSize, Qt, QThread, QThreadPool
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from ui.input_process import DebugInputProcessor
from utils import get_modifier_icon


class EnemyPreviewDialog(QDialog):
    def __init__(self, enemy: Enemy, parent=None):
        super().__init__(parent)
        self.enemy: Enemy = enemy
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self.setWindowIcon(QIcon(resource_path("assets/llmaker_logo.png")))
        self.setMinimumSize(QSize(400, 300))

        self.setup_ui()

    def setup_ui(self):

        self.setWindowTitle(f"Details for {self.enemy.name}")
        self.dialog_layout = QGridLayout(self)

        pixmap = QPixmap(os.path.join(config.entity.save_dir, self.enemy.sprite))

        sprite_label = QLabel()
        sprite_label.setPixmap(
            pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio)
        )

        # Enemy details

        details_layout = QVBoxLayout()

        description_label = QLabel(f"<b>Description</b>: {self.enemy.description}")
        description_label.setWordWrap(True)

        details_layout.addWidget(QLabel(f"<b>Name</b>: {self.enemy.name}"))
        details_layout.addWidget(description_label)
        details_layout.addWidget(QLabel(f"<b>Species</b>: {self.enemy.species}"))
        details_layout.addWidget(QLabel(f"<b>HP</b>: {self.enemy.hp}"))
        details_layout.addWidget(QLabel(f"<b>Dodge</b>: {self.enemy.dodge}"))
        details_layout.addWidget(QLabel(f"<b>Prot</b>: {self.enemy.prot}"))
        details_layout.addWidget(QLabel(f"<b>Spd</b>: {self.enemy.spd}"))

        # Layout for Attacks

        attacks_layout = QVBoxLayout()
        attacks_layout.addWidget(QLabel("<b>Attacks:</b>"))  # Section title

        # Create a scrollable area for attacks in case there are too many

        attacks_scroll_area = QScrollArea(parent=self)
        attacks_scroll_widget = QWidget(parent=self)
        attacks_scroll_widget.setProperty("attacks_grid", "yes")
        attacks_grid_layout = QGridLayout(attacks_scroll_widget)

        # Add headers for the grid

        attacks_grid_layout.addWidget(QLabel("<b>Name</b>"), 0, 0)
        attacks_grid_layout.addWidget(QLabel("<b>Description</b>"), 0, 1)
        attacks_grid_layout.addWidget(QLabel("<b>From</b>"), 0, 2)
        attacks_grid_layout.addWidget(QLabel("<b>To</b>"), 0, 3)
        attacks_grid_layout.addWidget(QLabel("<b>Base Damage</b>"), 0, 4)
        attacks_grid_layout.addWidget(QLabel("<b>Modifier</b>"), 0, 5)

        # Populate the grid with attacks

        for row, attack in enumerate(self.enemy.attacks, start=1):
            attack_description_label = QLabel(attack.description)
            attack_description_label.setWordWrap(True)

            attacks_grid_layout.addWidget(QLabel(attack.name), row, 0)
            attacks_grid_layout.addWidget(attack_description_label, row, 1)
            attacks_grid_layout.addWidget(QLabel(attack.starting_positions), row, 2)
            attacks_grid_layout.addWidget(QLabel(attack.target_positions), row, 3)
            attacks_grid_layout.addWidget(QLabel(str(attack.base_dmg)), row, 4)
            if attack.modifier is not None:
                icon = resource_path(
                    get_modifier_icon(
                        get_enum_by_value(ModifierType, attack.modifier.type)
                    )
                )
                widg = QLabel(f'<img src="{icon}" width="16" height="16">')
                widg.setToolTip(str(attack.modifier))
                attacks_grid_layout.addWidget(widg, row, 5)
            else:
                attacks_grid_layout.addWidget(QLabel("N/A"), row, 5)
        # Set up the scroll area for the attacks list

        attacks_scroll_area.setWidget(attacks_scroll_widget)
        attacks_scroll_area.setWidgetResizable(True)

        # Add the scroll area to the layout

        attacks_layout.addWidget(attacks_scroll_area)

        # Add a button box for OK button

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(self.reject)

        # Layout the dialog

        self.dialog_layout.addWidget(sprite_label, 0, 0)
        self.dialog_layout.addLayout(details_layout, 0, 1)
        self.dialog_layout.addLayout(attacks_layout, 1, 0, 1, 2)
        self.dialog_layout.addWidget(button_box, 2, 0, 1, 2)


class FocusWatcher(QObject):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.FocusIn:
            self.callback()
        return super().eventFilter(obj, event)


class UserModeDialog(QDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(parent)
        self.level: Level = level
        self.func: LibCommand = func

        self.setWindowIcon(QIcon(resource_path("assets/llmaker_logo.png")))

        self.layout = QVBoxLayout()

        # Add submit button

        self.submit_btn = QPushButton("Submit")
        self.submit_btn.clicked.connect(self.submit)

        self.threadpool = QThreadPool()

        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 100)
        self.pbar.setHidden(True)

        self.pause_last_time = time()
        self.pause_times = []

        self.setLayout(self.layout)

        self.focus_watcher = FocusWatcher(self.on_child_focus)

    def connect_child_signals(self) -> None:
        for c in self.parentWidget().findChildren((QLineEdit, QComboBox)):
            c.installEventFilter(self.focus_watcher)
            if hasattr(c, "currentTextChanged"):
                c.currentTextChanged.connect(self.on_value_changed)
            if hasattr(c, "textChanged"):
                c.textChanged.connect(self.on_value_changed)

    def on_value_changed(self) -> None:
        # while we are still editing values, we are not paused

        self.pause_last_time = time()

    def on_child_focus(self) -> None:
        time_diff = time() - self.pause_last_time
        self.pause_times.append(time_diff)
        self.pause_last_time = time()
        logging.getLogger("gui").debug(f"Time between edits: {time_diff:.2f}s")

    def update_progress(self, progress):
        logging.getLogger("gui").debug(
            f"{self.func.internal_name} - Task progress: {progress}"
        )
        self.pbar.setValue(progress)

    def task_success(self, result):
        logging.getLogger("gui").debug(f"{self.func.internal_name} - Edit finished")
        button_pressed = QMessageBox.information(self, "Output", f"{result}")
        if button_pressed == QMessageBox.StandardButton.Ok:
            self.close()

    def task_finished(self):
        self.pbar.reset()
        self.pbar.setHidden(True)
        self.submit_btn.setDisabled(False)

    def task_error(self, err_data):
        logging.getLogger("gui").error(
            f"LLMaker Error: {err_data[0].__name__} - {str(err_data[1])}"
        )
        QMessageBox.critical(
            self, f"LLMaker Error: {err_data[0].__name__}", str(err_data[1])
        )

    def get_kwargs(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def submit(self):
        worker = DebugInputProcessor(
            self.get_kwargs(),
            self.func,
        )
        worker.signals.result.connect(self.task_success)
        worker.signals.error.connect(self.task_error)
        worker.signals.progress.connect(self.update_progress)
        worker.signals.finished.connect(self.task_finished)

        self.pbar.setHidden(False)
        self.pbar.reset()

        self.submit_btn.setDisabled(True)

        logging.getLogger("gui").debug(
            f"Average pause duration: {(sum(self.pause_times) / len(self.pause_times)) if len(self.pause_times) > 0 else 0.0:.2f}s"
        )

        self.threadpool.start(worker)


class AddRoomDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Add Room")

        self.layout.addWidget(QLabel("Room name:"))
        self.roomname_widget = QLineEdit()
        self.layout.addWidget(self.roomname_widget)

        self.layout.addWidget(QLabel("Description"))
        self.roomdescription_widget = QLineEdit()
        self.layout.addWidget(self.roomdescription_widget)

        if len(self.level.rooms) != 0:
            self.rooms_combobox = QComboBox()
            self.rooms_combobox.addItems(list(self.level.rooms.keys()))
            self.rooms_combobox.setCurrentText(self.level.current_room)
            self.rooms_combobox.currentTextChanged.connect(self.room_from_changed)
            self.layout.addWidget(QLabel("Connecting room:"))
            self.layout.addWidget(self.rooms_combobox)

            self.directions_combobox = QComboBox()
            room_from = (
                self.level.current_room
                if self.level.current_room in self.level.rooms.keys()
                else self.level.corridors[self.level.current_room].room_from
            )
            valid_directions = [
                direction.value
                for direction in Direction
                if self.level.connections[room_from][direction] == ""
            ]
            self.directions_combobox.addItems(valid_directions)
            self.directions_combobox.setCurrentText(valid_directions[0])
            self.layout.addWidget(QLabel("Direction to:"))
            self.layout.addWidget(self.directions_combobox)
        self.connect_child_signals()

        self.submit_btn.setText("Add room")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

    def room_from_changed(self, room_name: str) -> None:
        self.directions_combobox.clear()
        valid_directions = [
            direction.value
            for direction in Direction
            if self.level.connections[room_name][direction] == ""
        ]
        self.directions_combobox.addItems(valid_directions)
        self.directions_combobox.setCurrentText(valid_directions[0])

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.text()
        room_description = self.roomdescription_widget.text()
        if len(self.level.rooms) != 0:
            other_room = self.rooms_combobox.currentText()
            direction = self.directions_combobox.currentText()
        else:
            other_room = ""
            direction = Direction.NORTH.value
        return {
            "self": None,
            "level": self.level,
            "name": room_name,
            "description": room_description,
            "room_from": other_room,
            "direction": direction,
        }


class RemoveRoomDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Remove Room")

        self.layout.addWidget(QLabel("Room name:"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.layout.addWidget(self.roomname_widget)

        self.submit_btn.setText("Remove room")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        return {
            "self": None,
            "level": self.level,
            "name": room_name,
        }


class UpdateRoomDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Update Room Properties")

        self.layout.addWidget(QLabel("Which room?"))
        self.roomrefname_widget = QComboBox()
        self.roomrefname_widget.addItems(list(self.level.rooms.keys()))
        self.roomrefname_widget.setCurrentText(self.level.current_room)
        self.roomrefname_widget.currentTextChanged.connect(self.refroom_changed)
        self.layout.addWidget(self.roomrefname_widget)

        self.layout.addWidget(QLabel("Room name:"))
        self.roomname_widget = QLineEdit()
        self.roomname_widget.setText(self.level.current_room)
        self.layout.addWidget(self.roomname_widget)

        self.layout.addWidget(QLabel("Description"))
        self.roomdescription_widget = QLineEdit()
        self.roomdescription_widget.setText(
            self.level.rooms[self.level.current_room].description
        )
        self.layout.addWidget(self.roomdescription_widget)

        self.submit_btn.setText("Update room")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

    def refroom_changed(self, room_ref_name: str) -> None:
        self.roomname_widget.setText(room_ref_name)
        self.roomdescription_widget.setText(self.level.rooms[room_ref_name].description)

    def get_kwargs(self) -> Dict[str, Any]:
        room_reference_name = self.roomrefname_widget.currentText()
        room_name = self.roomname_widget.text()
        room_description = self.roomdescription_widget.text()
        return {
            "self": None,
            "level": self.level,
            "room_reference_name": room_reference_name,
            "name": room_name,
            "description": room_description,
        }


class AddCorridorDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Add Corridor")

        self.layout.addWidget(QLabel("From which room?"))
        self.roomfrom_widget = QComboBox()
        self.roomfrom_widget.addItems(list(self.level.rooms.keys()))
        self.roomfrom_widget.setCurrentText(self.level.current_room)
        self.roomfrom_widget.currentTextChanged.connect(self.refroom_changed)
        self.layout.addWidget(self.roomfrom_widget)

        self.layout.addWidget(QLabel("To which room?"))
        self.roomto_widget = QComboBox()
        self.roomto_widget.addItems(list(self.level.rooms.keys()))
        self.roomto_widget.setCurrentText(list(self.level.rooms.keys())[0])
        self.layout.addWidget(self.roomto_widget)

        self.layout.addWidget(QLabel("Corridor length:"))
        self.corridorlen_widget = QSpinBox()
        self.corridorlen_widget.setValue(config.dungeon.corridor_min_length)
        self.corridorlen_widget.setMinimum(config.dungeon.corridor_min_length)
        self.corridorlen_widget.setMaximum(config.dungeon.corridor_max_length)
        self.layout.addWidget(self.corridorlen_widget)

        self.directions_combobox = QComboBox()
        valid_directions = [
            direction.value
            for direction in Direction
            if self.level.connections[self.level.current_room][direction] == ""
        ]
        self.directions_combobox.addItems(valid_directions)
        self.directions_combobox.setCurrentText(valid_directions[0])
        self.layout.addWidget(QLabel("To which direction?:"))
        self.layout.addWidget(self.directions_combobox)

        self.submit_btn.setText("Add corridor")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

    def refroom_changed(self, room_name: str) -> None:
        self.directions_combobox.clear()
        valid_directions = [
            direction.value
            for direction in Direction
            if self.level.connections[room_name][direction] == ""
        ]
        self.directions_combobox.addItems(valid_directions)
        self.directions_combobox.setCurrentText(valid_directions[0])

    def get_kwargs(self) -> Dict[str, Any]:
        room_from_name = self.roomfrom_widget.currentText()
        room_to_name = self.roomto_widget.currentText()
        corridor_length = self.corridorlen_widget.value()
        direction = self.directions_combobox.currentText()
        return {
            "self": None,
            "level": self.level,
            "room_from_name": room_from_name,
            "room_to_name": room_to_name,
            "corridor_length": corridor_length,
            "direction": direction,
        }


class RemoveCorridorDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Remove Corridor")

        self.layout.addWidget(QLabel("Which corridor?"))
        self.corridors_widget = QComboBox()
        self.corridors_widget.addItems(list(self.level.corridors.keys()))
        self.layout.addWidget(self.corridors_widget)

        self.submit_btn.setText("Remove corridor")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

    def get_kwargs(self) -> Dict[str, Any]:
        corridor = self.level.corridors[self.corridors_widget.currentText()]
        room_from_name = corridor.room_from
        room_to_name = corridor.room_to
        return {
            "self": None,
            "level": self.level,
            "room_from_name": room_from_name,
            "room_to_name": room_to_name,
        }


class UpdateCorridorDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Update Corridor")

        if self.level.current_room in self.level.corridors.keys():
            ref_corridor = self.level.corridors[self.level.current_room]
        else:
            ref_corridor = list(self.level.corridors.values())[0]
        self.layout.addWidget(QLabel("Which corridor?"))
        self.corridors_widget = QComboBox()
        self.corridors_widget.addItems(list(self.level.corridors.keys()))
        self.corridors_widget.setCurrentText(ref_corridor.name)
        self.corridors_widget.currentTextChanged.connect(self.corridor_changed)
        self.layout.addWidget(self.corridors_widget)

        self.layout.addWidget(QLabel("From which room?"))
        self.roomfrom_widget = QComboBox()
        self.roomfrom_widget.addItems(list(self.level.rooms.keys()))
        self.roomfrom_widget.setCurrentText(ref_corridor.room_from)
        self.roomfrom_widget.currentTextChanged.connect(self.roomfrom_changed)
        self.layout.addWidget(self.roomfrom_widget)

        self.layout.addWidget(QLabel("To which room?"))
        self.roomto_widget = QComboBox()
        self.roomto_widget.addItems(list(self.level.rooms.keys()))
        self.roomto_widget.setCurrentText(ref_corridor.room_to)
        self.layout.addWidget(self.roomto_widget)

        self.layout.addWidget(QLabel("Corridor length:"))
        self.corridorlen_widget = QSpinBox()
        self.corridorlen_widget.setValue(ref_corridor.length)
        self.corridorlen_widget.setMinimum(config.dungeon.corridor_min_length)
        self.corridorlen_widget.setMaximum(config.dungeon.corridor_max_length)
        self.layout.addWidget(self.corridorlen_widget)

        self.directions_combobox = QComboBox()
        valid_directions = [
            direction.value
            for direction in Direction
            if self.level.connections[ref_corridor.room_from][direction] == ""
            or self.level.connections[ref_corridor.room_from][direction]
            == ref_corridor.room_to
        ]
        self.directions_combobox.addItems(valid_directions)
        self.directions_combobox.setCurrentText(ref_corridor.direction)
        self.layout.addWidget(QLabel("To which direction?:"))
        self.layout.addWidget(self.directions_combobox)

        self.submit_btn.setText("Update corridor")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

    def corridor_changed(self, corridor_name: str) -> None:
        corridor = self.level.corridors[corridor_name]
        self.roomfrom_widget.setCurrentText(corridor.room_from)
        self.roomto_widget.setCurrentText(corridor.room_to)
        self.corridorlen_widget.setValue(corridor.length)
        self.directions_combobox.clear()
        valid_directions = [
            direction.value
            for direction in Direction
            if self.level.connections[corridor.room_from][direction] == ""
            or self.level.connections[corridor.room_from][direction] == corridor.room_to
        ]
        self.directions_combobox.addItems(valid_directions)
        self.directions_combobox.setCurrentText(valid_directions[0])

    def roomfrom_changed(self, roomfrom_name: str) -> None:
        self.directions_combobox.clear()
        valid_directions = [
            direction.value
            for direction in Direction
            if self.level.connections[roomfrom_name][direction] == ""
        ]
        self.directions_combobox.addItems(valid_directions)
        self.directions_combobox.setCurrentText(valid_directions[0])

    def get_kwargs(self) -> Dict[str, Any]:
        corridor = self.level.corridors[self.corridors_widget.currentText()]
        room_from_reference_name = corridor.room_from
        room_to_reference_name = corridor.room_to
        room_from_name = self.roomfrom_widget.currentText()
        room_to_name = self.roomto_widget.currentText()
        corridor_length = self.corridorlen_widget.value()
        direction = self.directions_combobox.currentText()
        return {
            "self": None,
            "level": self.level,
            "room_from_reference_name": room_from_reference_name,
            "room_to_reference_name": room_to_reference_name,
            "room_from_name": room_from_name,
            "room_to_name": room_to_name,
            "corridor_length": corridor_length,
            "direction": direction,
        }


class AddEntityDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Add Entity")

        self.layout.addWidget(QLabel("Which type?"))
        self.type_widget = QComboBox()
        self.type_widget.addItems([t.value for t in EntityEnum])
        self.type_widget.setCurrentText(EntityEnum.ENEMY.value)
        self.type_widget.currentTextChanged.connect(self.type_changed)
        self.layout.addWidget(self.type_widget)

        self.layout.addWidget(QLabel("To which room/corridor?"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.addItems(list(self.level.corridors.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.roomname_widget.currentTextChanged.connect(self.roomname_changed)
        self.layout.addWidget(self.roomname_widget)

        self.corridorcell_container = QWidget(parent=self)
        corridorcell_layout = QVBoxLayout(self.corridorcell_container)
        corridorcell_layout.addWidget(QLabel("In which corridor cell?"))
        self.corridorcell_widget = QSpinBox()
        self.corridorcell_widget.setValue(1)
        self.corridorcell_widget.setMinimum(1)
        self.corridorcell_widget.setMaximum(config.dungeon.corridor_max_length)
        corridorcell_layout.addWidget(self.corridorcell_widget)
        self.layout.addWidget(self.corridorcell_container)

        self.layout.addWidget(QLabel("Name:"))
        self.name_widget = QLineEdit()
        self.layout.addWidget(self.name_widget)

        self.layout.addWidget(QLabel("Description"))
        self.description_widget = QLineEdit()
        self.layout.addWidget(self.description_widget)

        # enemy properties

        self.species_container = QWidget(parent=self)
        species_layout = QVBoxLayout(self.species_container)
        species_layout.addWidget(QLabel("Species:"))
        self.species_widget = QLineEdit()
        species_layout.addWidget(self.species_widget)
        self.layout.addWidget(self.species_container)

        self.hp_container = QWidget(parent=self)
        hp_layout = QHBoxLayout(self.hp_container)
        hp_layout.addWidget(QLabel("HP:"))
        self.hp_widget = QDoubleSpinBox()
        self.hp_widget.setSingleStep(0.01)
        self.hp_widget.setValue(config.dungeon.min_hp)
        self.hp_widget.setMinimum(config.dungeon.min_hp)
        self.hp_widget.setMaximum(config.dungeon.max_hp)
        hp_layout.addWidget(self.hp_widget)
        self.layout.addWidget(self.hp_container)

        self.dodge_container = QWidget(parent=self)
        dodge_layout = QHBoxLayout(self.dodge_container)
        dodge_layout.addWidget(QLabel("Dodge:"))
        self.dodge_widget = QDoubleSpinBox()
        self.dodge_widget.setSingleStep(0.01)
        self.dodge_widget.setValue(config.dungeon.min_dodge)
        self.dodge_widget.setMinimum(config.dungeon.min_dodge)
        self.dodge_widget.setMaximum(config.dungeon.max_dodge)
        dodge_layout.addWidget(self.dodge_widget)
        self.layout.addWidget(self.dodge_container)

        self.prot_container = QWidget(parent=self)
        prot_layout = QHBoxLayout(self.prot_container)
        prot_layout.addWidget(QLabel("Protection:"))
        self.prot_widget = QDoubleSpinBox()
        self.prot_widget.setSingleStep(0.01)
        self.prot_widget.setValue(config.dungeon.min_prot)
        self.prot_widget.setMinimum(config.dungeon.min_prot)
        self.prot_widget.setMaximum(config.dungeon.max_prot)
        prot_layout.addWidget(self.prot_widget)
        self.layout.addWidget(self.prot_container)

        self.spd_container = QWidget(parent=self)
        spd_layout = QHBoxLayout(self.spd_container)
        spd_layout.addWidget(QLabel("Speed:"))
        self.spd_widget = QDoubleSpinBox()
        self.spd_widget.setSingleStep(0.01)
        self.spd_widget.setValue(config.dungeon.min_spd)
        self.spd_widget.setMinimum(config.dungeon.min_spd)
        self.spd_widget.setMaximum(config.dungeon.max_spd)
        spd_layout.addWidget(self.spd_widget)
        self.layout.addWidget(self.spd_container)

        # trap properties

        self.effect_container = QWidget(parent=self)
        effect_layout = QVBoxLayout(self.effect_container)
        effect_layout.addWidget(QLabel("Effect:"))
        self.effect_widget = QLineEdit()
        effect_layout.addWidget(self.effect_widget)
        self.layout.addWidget(self.effect_container)

        # treasure properties

        self.loot_container = QWidget(parent=self)
        loot_layout = QVBoxLayout(self.loot_container)
        loot_layout.addWidget(QLabel("Loot"))
        self.loot_widget = QLineEdit()
        loot_layout.addWidget(self.loot_widget)
        self.layout.addWidget(self.loot_container)

        self.trappedchance_container = QWidget(parent=self)
        trappedchance_layout = QHBoxLayout(self.trappedchance_container)
        trappedchance_layout.addWidget(QLabel("Trapped Chance:"))
        self.trappedchance_widget = QDoubleSpinBox()
        self.trappedchance_widget.setSingleStep(0.01)
        self.trappedchance_widget.setValue(0.0)
        self.trappedchance_widget.setMinimum(0.0)
        self.trappedchance_widget.setMaximum(1.0)
        trappedchance_layout.addWidget(self.trappedchance_widget)
        self.layout.addWidget(self.trappedchance_container)

        self.dmg_container = QWidget(parent=self)
        dmg_layout = QHBoxLayout(self.dmg_container)
        dmg_layout.addWidget(QLabel("Damage:"))
        self.dmg_widget = QDoubleSpinBox()
        self.dmg_widget.setSingleStep(0.01)
        self.dmg_widget.setValue(config.dungeon.min_base_dmg)
        self.dmg_widget.setMinimum(config.dungeon.min_base_dmg)
        self.dmg_widget.setMaximum(config.dungeon.max_base_dmg)
        dmg_layout.addWidget(self.dmg_widget)
        self.layout.addWidget(self.dmg_container)

        self.modifier_container = QWidget(parent=self)
        modifier_layout = QHBoxLayout(self.modifier_container)
        modifier_layout.addWidget(QLabel("Modifier:"))
        self.modifier_widget = QComboBox()
        self.modifier_widget.addItems(["None"] + [x.value for x in ModifierType])
        self.modifier_widget.setCurrentText("None")
        self.modifier_widget.currentTextChanged.connect(self.modifiertype_changed)
        modifier_layout.addWidget(self.modifier_widget)
        self.layout.addWidget(self.modifier_container)

        self.modifierchance_container = QWidget(parent=self)
        modifierchance_layout = QHBoxLayout(self.modifierchance_container)
        modifierchance_layout.addWidget(QLabel("Modifier Chance:"))
        self.modifierchance_widget = QDoubleSpinBox()
        self.modifierchance_widget.setSingleStep(0.01)
        self.modifierchance_widget.setValue(0.0)
        self.modifierchance_widget.setMinimum(0.0)
        self.modifierchance_widget.setMaximum(1.0)
        modifierchance_layout.addWidget(self.modifierchance_widget)
        self.layout.addWidget(self.modifierchance_container)

        self.modifierturns_container = QWidget(parent=self)
        modifierturns_layout = QHBoxLayout(self.modifierturns_container)
        modifierturns_layout.addWidget(QLabel("Modifier Turns:"))
        self.modifierturns_widget = QSpinBox()
        self.modifierturns_widget.setSingleStep(1)
        self.modifierturns_widget.setValue(0)
        self.modifierturns_widget.setMinimum(0)
        self.modifierturns_widget.setMaximum(5)
        modifierturns_layout.addWidget(self.modifierturns_widget)
        self.layout.addWidget(self.modifierturns_container)

        self.modifieramount_container = QWidget(parent=self)
        modifieramount_layout = QHBoxLayout(self.modifieramount_container)
        modifieramount_layout.addWidget(QLabel("Modifier Amount:"))
        self.modifieramount_widget = QDoubleSpinBox()
        self.modifieramount_widget.setSingleStep(0.01)
        self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
        modifieramount_layout.addWidget(self.modifieramount_widget)
        self.layout.addWidget(self.modifieramount_container)

        self.submit_btn.setText("Add enemy")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

        self.type_changed(EntityEnum.ENEMY.value)
        self.roomname_changed(self.level.current_room)
        self.modifiertype_changed("None")

    def type_changed(self, t: str) -> None:
        t_enum = get_enum_by_value(EntityEnum, t)
        if t_enum == EntityEnum.ENEMY:
            suffix = "enemy"

            self.species_container.show()
            self.hp_container.show()
            self.dodge_container.show()
            self.prot_container.show()
            self.spd_container.show()
            self.effect_container.hide()
            self.loot_container.hide()
            self.trappedchance_container.hide()
            self.dmg_container.hide()
            self.modifier_container.hide()
            self.modifierchance_container.hide()
            self.modifierturns_container.hide()
            self.modifieramount_container.hide()

            self.roomname_widget.clear()
            self.roomname_widget.addItems(list(self.level.rooms.keys()))
            self.roomname_widget.addItems(list(self.level.corridors.keys()))
            self.roomname_widget.setCurrentText(self.level.current_room)
        elif t_enum == EntityEnum.TRAP:
            suffix = "trap"

            self.species_container.hide()
            self.hp_container.hide()
            self.dodge_container.hide()
            self.prot_container.hide()
            self.spd_container.hide()
            self.effect_container.show()
            self.loot_container.hide()
            self.trappedchance_container.show()
            self.dmg_container.show()
            self.modifier_container.show()
            self.modifierchance_container.show()
            self.modifierturns_container.show()
            self.modifieramount_container.show()

            self.roomname_widget.clear()
            self.roomname_widget.addItems(list(self.level.corridors.keys()))
            if self.level.current_room in self.level.corridors.keys():
                self.roomname_widget.setCurrentText(self.level.current_room)
            else:
                self.roomname_widget.setCurrentText(
                    list(self.level.corridors.keys())[0]
                )
            self.trappedchance_container.children()[1].setText("Chance:")

            self.modifiertype_changed(self.modifier_widget.currentText())
        else:
            suffix = "treasure"

            self.species_container.hide()
            self.hp_container.hide()
            self.dodge_container.hide()
            self.prot_container.hide()
            self.spd_container.hide()
            self.effect_container.hide()
            self.loot_container.show()
            self.trappedchance_container.show()
            self.dmg_container.show()
            self.modifier_container.show()
            self.modifierchance_container.show()
            self.modifierturns_container.show()
            self.modifieramount_container.show()

            self.roomname_widget.clear()
            self.roomname_widget.addItems(list(self.level.rooms.keys()))
            self.roomname_widget.addItems(list(self.level.corridors.keys()))
            self.roomname_widget.setCurrentText(self.level.current_room)

            self.trappedchance_container.children()[1].setText("Trapped Chance:")

            self.modifiertype_changed(self.modifier_widget.currentText())
        self.func = DungeonCrawlerFunctions().FunctionDict[f"add_{suffix}"]
        self.submit_btn.setText(f"Add {suffix}")

    def roomname_changed(self, roomname: str) -> None:
        if roomname in self.level.rooms.keys():
            self.corridorcell_container.hide()
        elif roomname in self.level.corridors.keys():
            self.corridorcell_container.show()

            corridor = self.level.corridors[roomname]
            self.corridorcell_widget.setMaximum(corridor.length)
        else:  # triggered on .clear()
            pass

    def modifiertype_changed(self, modifiertype: str) -> None:
        if modifiertype == "None":
            self.modifierchance_container.hide()
            self.modifierturns_container.hide()
            self.modifieramount_container.hide()
        else:
            self.modifierchance_container.show()
            self.modifierturns_container.show()

            m_type = get_enum_by_value(ModifierType, modifiertype)

            if m_type == ModifierType.BLEED or m_type == ModifierType.HEAL:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
            elif m_type == ModifierType.SCARE:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(0.0)
                self.modifieramount_widget.setMinimum(0.0)
                self.modifieramount_widget.setMaximum(1.0)
            else:  # m_type is STUN
                self.modifieramount_container.hide()

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        cell_index = (
            self.corridorcell_widget.value()
            if room_name in self.level.corridors.keys()
            else -1
        )
        t_enum = get_enum_by_value(EntityEnum, self.type_widget.currentText())
        if t_enum == EntityEnum.ENEMY:
            return {
                "self": None,
                "level": self.level,
                "room_name": room_name,
                "cell_index": cell_index,
                "name": self.name_widget.text(),
                "description": self.description_widget.text(),
                "species": self.species_widget.text(),
                "hp": self.hp_widget.value(),
                "dodge": self.dodge_widget.value(),
                "prot": self.prot_widget.value(),
                "spd": self.spd_widget.value(),
            }
        elif t_enum == EntityEnum.TRAP:
            m_str = self.modifier_widget.currentText()
            if m_str == "None":
                m_str = "no-modifier"
            return {
                "self": None,
                "level": self.level,
                "corridor_name": room_name,
                "cell_index": cell_index,
                "name": self.name_widget.text(),
                "description": self.description_widget.text(),
                "effect": self.effect_widget.text(),
                "chance": self.trappedchance_widget.value(),
                "dmg": self.dmg_widget.value(),
                "modifier_type": m_str,
                "modifier_chance": self.modifierchance_widget.value(),
                "modifier_turns": self.modifierturns_widget.value(),
                "modifier_amount": self.modifieramount_widget.value(),
            }
        else:  # t_enum is EntityEnum.TREASURE
            m_str = self.modifier_widget.currentText()
            if m_str == "None":
                m_str = "no-modifier"
            return {
                "self": None,
                "level": self.level,
                "room_name": room_name,
                "cell_index": cell_index,
                "name": self.name_widget.text(),
                "description": self.description_widget.text(),
                "loot": self.loot_widget.text(),
                "trapped_chance": self.trappedchance_widget.value(),
                "dmg": self.dmg_widget.value(),
                "modifier_type": m_str,
                "modifier_chance": self.modifierchance_widget.value(),
                "modifier_turns": self.modifierturns_widget.value(),
                "modifier_amount": self.modifieramount_widget.value(),
            }


class UpdateEntityDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Edit Entity")

        self.layout.addWidget(QLabel("Which type?"))
        self.type_widget = QComboBox()
        self.type_widget.addItems([t.value for t in EntityEnum])
        self.type_widget.setCurrentText(EntityEnum.ENEMY.value)
        self.type_widget.currentTextChanged.connect(self.type_changed)
        self.layout.addWidget(self.type_widget)

        self.layout.addWidget(QLabel("In which room/corridor?"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.addItems(list(self.level.corridors.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.roomname_widget.currentTextChanged.connect(self.roomname_changed)
        self.layout.addWidget(self.roomname_widget)

        self.corridorcell_container = QWidget(parent=self)
        corridorcell_layout = QVBoxLayout(self.corridorcell_container)
        corridorcell_layout.addWidget(QLabel("In which corridor cell?"))
        self.corridorcell_widget = QSpinBox()
        self.corridorcell_widget.setValue(1)
        self.corridorcell_widget.setMinimum(1)
        self.corridorcell_widget.setMaximum(config.dungeon.corridor_max_length)
        self.corridorcell_widget.valueChanged.connect(self.corridorcell_changed)
        corridorcell_layout.addWidget(self.corridorcell_widget)
        self.layout.addWidget(self.corridorcell_container)

        self.name_label = QLabel(f"Which {EntityEnum.ENEMY.value}?")
        self.layout.addWidget(self.name_label)
        self.ref_name_widget = QComboBox()
        self.ref_name_widget.currentTextChanged.connect(self.refname_changed)
        self.layout.addWidget(self.ref_name_widget)

        self.layout.addWidget(QLabel("Name:"))
        self.name_widget = QLineEdit()
        self.layout.addWidget(self.name_widget)

        self.layout.addWidget(QLabel("Description"))
        self.description_widget = QLineEdit()
        self.layout.addWidget(self.description_widget)

        # enemy properties

        self.species_container = QWidget(parent=self)
        species_layout = QVBoxLayout(self.species_container)
        species_layout.addWidget(QLabel("Species:"))
        self.species_widget = QLineEdit()
        species_layout.addWidget(self.species_widget)
        self.layout.addWidget(self.species_container)

        self.hp_container = QWidget(parent=self)
        hp_layout = QHBoxLayout(self.hp_container)
        hp_layout.addWidget(QLabel("HP:"))
        self.hp_widget = QDoubleSpinBox()
        self.hp_widget.setSingleStep(0.01)
        self.hp_widget.setValue(config.dungeon.min_hp)
        self.hp_widget.setMinimum(config.dungeon.min_hp)
        self.hp_widget.setMaximum(config.dungeon.max_hp)
        hp_layout.addWidget(self.hp_widget)
        self.layout.addWidget(self.hp_container)

        self.dodge_container = QWidget(parent=self)
        dodge_layout = QHBoxLayout(self.dodge_container)
        dodge_layout.addWidget(QLabel("Dodge:"))
        self.dodge_widget = QDoubleSpinBox()
        self.dodge_widget.setSingleStep(0.01)
        self.dodge_widget.setValue(config.dungeon.min_dodge)
        self.dodge_widget.setMinimum(config.dungeon.min_dodge)
        self.dodge_widget.setMaximum(config.dungeon.max_dodge)
        dodge_layout.addWidget(self.dodge_widget)
        self.layout.addWidget(self.dodge_container)

        self.prot_container = QWidget(parent=self)
        prot_layout = QHBoxLayout(self.prot_container)
        prot_layout.addWidget(QLabel("Protection:"))
        self.prot_widget = QDoubleSpinBox()
        self.prot_widget.setSingleStep(0.01)
        self.prot_widget.setValue(config.dungeon.min_prot)
        self.prot_widget.setMinimum(config.dungeon.min_prot)
        self.prot_widget.setMaximum(config.dungeon.max_prot)
        prot_layout.addWidget(self.prot_widget)
        self.layout.addWidget(self.prot_container)

        self.spd_container = QWidget(parent=self)
        spd_layout = QHBoxLayout(self.spd_container)
        spd_layout.addWidget(QLabel("Speed:"))
        self.spd_widget = QDoubleSpinBox()
        self.spd_widget.setSingleStep(0.01)
        self.spd_widget.setValue(config.dungeon.min_spd)
        self.spd_widget.setMinimum(config.dungeon.min_spd)
        self.spd_widget.setMaximum(config.dungeon.max_spd)
        spd_layout.addWidget(self.spd_widget)
        self.layout.addWidget(self.spd_container)

        # trap properties

        self.effect_container = QWidget(parent=self)
        effect_layout = QVBoxLayout(self.effect_container)
        effect_layout.addWidget(QLabel("Effect:"))
        self.effect_widget = QLineEdit()
        effect_layout.addWidget(self.effect_widget)
        self.layout.addWidget(self.effect_container)

        # treasure properties

        self.loot_container = QWidget(parent=self)
        loot_layout = QVBoxLayout(self.loot_container)
        loot_layout.addWidget(QLabel("Loot"))
        self.loot_widget = QLineEdit()
        loot_layout.addWidget(self.loot_widget)
        self.layout.addWidget(self.loot_container)

        self.trappedchance_container = QWidget(parent=self)
        trappedchance_layout = QHBoxLayout(self.trappedchance_container)
        trappedchance_layout.addWidget(QLabel("Trapped Chance:"))
        self.trappedchance_widget = QDoubleSpinBox()
        self.trappedchance_widget.setSingleStep(0.01)
        self.trappedchance_widget.setValue(0.0)
        self.trappedchance_widget.setMinimum(0.0)
        self.trappedchance_widget.setMaximum(1.0)
        trappedchance_layout.addWidget(self.trappedchance_widget)
        self.layout.addWidget(self.trappedchance_container)

        self.dmg_container = QWidget(parent=self)
        dmg_layout = QHBoxLayout(self.dmg_container)
        dmg_layout.addWidget(QLabel("Damage:"))
        self.dmg_widget = QDoubleSpinBox()
        self.dmg_widget.setSingleStep(0.01)
        self.dmg_widget.setValue(config.dungeon.min_base_dmg)
        self.dmg_widget.setMinimum(config.dungeon.min_base_dmg)
        self.dmg_widget.setMaximum(config.dungeon.max_base_dmg)
        dmg_layout.addWidget(self.dmg_widget)
        self.layout.addWidget(self.dmg_container)

        self.modifier_container = QWidget(parent=self)
        modifier_layout = QHBoxLayout(self.modifier_container)
        modifier_layout.addWidget(QLabel("Modifier:"))
        self.modifier_widget = QComboBox()
        self.modifier_widget.addItems(["None"] + [x.value for x in ModifierType])
        self.modifier_widget.setCurrentText("None")
        self.modifier_widget.currentTextChanged.connect(self.modifiertype_changed)
        modifier_layout.addWidget(self.modifier_widget)
        self.layout.addWidget(self.modifier_container)

        self.modifierchance_container = QWidget(parent=self)
        modifierchance_layout = QHBoxLayout(self.modifierchance_container)
        modifierchance_layout.addWidget(QLabel("Modifier Chance:"))
        self.modifierchance_widget = QDoubleSpinBox()
        self.modifierchance_widget.setSingleStep(0.01)
        self.modifierchance_widget.setValue(0.0)
        self.modifierchance_widget.setMinimum(0.0)
        self.modifierchance_widget.setMaximum(1.0)
        modifierchance_layout.addWidget(self.modifierchance_widget)
        self.layout.addWidget(self.modifierchance_container)

        self.modifierturns_container = QWidget(parent=self)
        modifierturns_layout = QHBoxLayout(self.modifierturns_container)
        modifierturns_layout.addWidget(QLabel("Modifier Turns:"))
        self.modifierturns_widget = QSpinBox()
        self.modifierturns_widget.setSingleStep(1)
        self.modifierturns_widget.setValue(0)
        self.modifierturns_widget.setMinimum(0)
        self.modifierturns_widget.setMaximum(5)
        modifierturns_layout.addWidget(self.modifierturns_widget)
        self.layout.addWidget(self.modifierturns_container)

        self.modifieramount_container = QWidget(parent=self)
        modifieramount_layout = QHBoxLayout(self.modifieramount_container)
        modifieramount_layout.addWidget(QLabel("Modifier Amount:"))
        self.modifieramount_widget = QDoubleSpinBox()
        self.modifieramount_widget.setSingleStep(0.01)
        self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
        modifieramount_layout.addWidget(self.modifieramount_widget)
        self.layout.addWidget(self.modifieramount_container)

        self.submit_btn.setText("Update enemy")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

        self.roomname_changed(self.level.current_room)
        self.type_changed(EntityEnum.ENEMY.value)

    def type_changed(self, t: str) -> None:
        t_enum = get_enum_by_value(EntityEnum, t)
        if t_enum == EntityEnum.ENEMY:
            self.species_container.show()
            self.hp_container.show()
            self.dodge_container.show()
            self.prot_container.show()
            self.spd_container.show()
            self.effect_container.hide()
            self.loot_container.hide()
            self.trappedchance_container.hide()
            self.dmg_container.hide()
            self.modifier_container.hide()
            self.modifierchance_container.hide()
            self.modifierturns_container.hide()
            self.modifieramount_container.hide()
        elif t_enum == EntityEnum.TRAP:
            self.species_container.hide()
            self.hp_container.hide()
            self.dodge_container.hide()
            self.prot_container.hide()
            self.spd_container.hide()
            self.effect_container.show()
            self.loot_container.hide()
            self.trappedchance_container.show()
            self.dmg_container.show()
            self.modifier_container.show()
            self.modifierchance_container.show()
            self.modifierturns_container.show()
            self.modifieramount_container.show()

            self.trappedchance_container.children()[1].setText("Chance:")
            self.modifiertype_changed(self.modifier_widget.currentText())
        else:
            self.species_container.hide()
            self.hp_container.hide()
            self.dodge_container.hide()
            self.prot_container.hide()
            self.spd_container.hide()
            self.effect_container.hide()
            self.loot_container.show()
            self.trappedchance_container.show()
            self.dmg_container.show()
            self.modifier_container.show()
            self.modifierchance_container.show()
            self.modifierturns_container.show()
            self.modifieramount_container.show()

            self.trappedchance_container.children()[1].setText("Trapped Chance:")

            self.modifiertype_changed(self.modifier_widget.currentText())
        self.ref_name_widget.clear()
        self.ref_name_widget.addItems(
            [x.name for x in self.get_encounter().entities[t]]
        )

        self.func = DungeonCrawlerFunctions().FunctionDict[f"update_{t}_properties"]
        self.submit_btn.setText(f"Update {t}")

    def roomname_changed(self, roomname: str) -> None:
        if roomname in self.level.rooms.keys():
            self.corridorcell_container.hide()
        else:
            self.corridorcell_container.show()
        self.ref_name_widget.clear()
        self.ref_name_widget.addItems(
            [
                x.name
                for x in self.get_encounter().entities[self.type_widget.currentText()]
            ]
        )

    def corridorcell_changed(self, cell_index: int) -> None:
        self.ref_name_widget.clear()
        self.ref_name_widget.addItems(
            [
                x.name
                for x in self.get_encounter().entities[self.type_widget.currentText()]
            ]
        )

    def modifiertype_changed(self, modifiertype: str) -> None:
        if modifiertype == "None":
            self.modifierchance_container.hide()
            self.modifierturns_container.hide()
            self.modifieramount_container.hide()
        else:
            self.modifierchance_container.show()
            self.modifierturns_container.show()

            m_type = get_enum_by_value(ModifierType, modifiertype)

            if m_type == ModifierType.BLEED or m_type == ModifierType.HEAL:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
            elif m_type == ModifierType.SCARE:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(0.0)
                self.modifieramount_widget.setMinimum(0.0)
                self.modifieramount_widget.setMaximum(1.0)
            else:  # m_type is STUN
                self.modifieramount_container.hide()
        self.ref_name_widget.clear()
        self.ref_name_widget.addItems(
            [
                x.name
                for x in self.get_encounter().entities[self.type_widget.currentText()]
            ]
        )

    def get_encounter(self) -> Encounter:
        if self.roomname_widget.currentText() in self.level.rooms.keys():
            encounter = self.level.rooms[self.roomname_widget.currentText()].encounter
        else:
            encounter = self.level.corridors[
                self.roomname_widget.currentText()
            ].encounters[self.corridorcell_widget.value() - 1]
        return encounter

    def refname_changed(self, refname: str) -> None:
        t_enum = get_enum_by_value(EntityEnum, self.type_widget.currentText())
        encounter = self.get_encounter()
        entity = encounter.get_entity_by_name(entity_type=t_enum, entity_name=refname)

        if entity is not None:
            self.submit_btn.setDisabled(False)
            self.name_widget.setText(entity.name)
            self.description_widget.setText(entity.description)

            if t_enum == EntityEnum.ENEMY:
                self.species_widget.setText(entity.species)
                self.hp_widget.setValue(entity.hp)
                self.dodge_widget.setValue(entity.dodge)
                self.prot_widget.setValue(entity.prot)
                self.spd_widget.setValue(entity.spd)
            else:
                self.trappedchance_widget.setValue(entity.trapped_chance)
                self.dmg_widget.setValue(entity.dmg)
                if entity.modifier is not None:
                    self.modifier_widget.setCurrentText(entity.modifier.type)
                    self.modifierchance_widget.setValue(entity.modifier.chance)
                    self.modifieramount_widget.setValue(entity.modifier.amount)
                    self.modifierturns_widget.setValue(entity.modifier.turns)
                elif t_enum == EntityEnum.TRAP:
                    self.effect_widget.setText(entity.effect)
                else:  # t_enum is EntityEnum.TREASURE
                    self.loot_widget.setText(entity.loot)
        else:
            self.submit_btn.setDisabled(True)
            self.name_widget.setText("")
            self.description_widget.setText("")
            self.species_widget.setText("")
            self.hp_widget.setValue(config.dungeon.min_hp)
            self.dodge_widget.setValue(config.dungeon.min_dodge)
            self.prot_widget.setValue(config.dungeon.min_prot)
            self.spd_widget.setValue(config.dungeon.min_spd)
            self.modifier_widget.setCurrentText("None")
            self.modifierchance_widget.setValue(0.0)
            self.modifieramount_widget.setValue(0.0)
            self.modifierturns_widget.setValue(0)
            self.effect_widget.setText("")
            self.trappedchance_widget.setValue(0.0)
            self.dmg_widget.setValue(config.dungeon.min_base_dmg)
            self.loot_widget.setText("")

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        cell_index = (
            self.corridorcell_widget.value()
            if room_name in self.level.corridors.keys()
            else -1
        )
        t_enum = get_enum_by_value(EntityEnum, self.type_widget.currentText())
        if t_enum == EntityEnum.ENEMY:
            return {
                "self": None,
                "level": self.level,
                "room_name": room_name,
                "cell_index": cell_index,
                "reference_name": self.ref_name_widget.currentText(),
                "name": self.name_widget.text(),
                "description": self.description_widget.text(),
                "species": self.species_widget.text(),
                "hp": self.hp_widget.value(),
                "dodge": self.dodge_widget.value(),
                "prot": self.prot_widget.value(),
                "spd": self.spd_widget.value(),
            }
        elif t_enum == EntityEnum.TRAP:
            m_str = self.modifier_widget.currentText()
            if m_str == "None":
                m_str = "no-modifier"
            return {
                "self": None,
                "level": self.level,
                "corridor_name": room_name,
                "cell_index": cell_index,
                "reference_name": self.ref_name_widget.currentText(),
                "name": self.name_widget.text(),
                "description": self.description_widget.text(),
                "effect": self.effect_widget.text(),
                "chance": self.trappedchance_widget.value(),
                "dmg": self.dmg_widget.value(),
                "modifier_type": m_str,
                "modifier_chance": self.modifierchance_widget.value(),
                "modifier_turns": self.modifierturns_widget.value(),
                "modifier_amount": self.modifieramount_widget.value(),
            }
        else:  # t_enum is EntityEnum.TREASURE
            m_str = self.modifier_widget.currentText()
            if m_str == "None":
                m_str = "no-modifier"
            return {
                "self": None,
                "level": self.level,
                "room_name": room_name,
                "cell_index": cell_index,
                "reference_name": self.ref_name_widget.currentText(),
                "name": self.name_widget.text(),
                "description": self.description_widget.text(),
                "loot": self.loot_widget.text(),
                "trapped_chance": self.trappedchance_widget.value(),
                "dmg": self.dmg_widget.value(),
                "modifier_type": m_str,
                "modifier_chance": self.modifierchance_widget.value(),
                "modifier_turns": self.modifierturns_widget.value(),
                "modifier_amount": self.modifieramount_widget.value(),
            }


class RemoveEntityDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Remove Entity")
        self.func = DungeonCrawlerFunctions().FunctionDict["remove_entity"]

        self.layout.addWidget(QLabel("In which room/corridor?"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.addItems(list(self.level.corridors.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.roomname_widget.currentTextChanged.connect(self.roomname_changed)
        self.layout.addWidget(self.roomname_widget)

        self.corridorcell_container = QWidget(parent=self)
        corridorcell_layout = QVBoxLayout(self.corridorcell_container)
        corridorcell_layout.addWidget(QLabel("In which corridor cell?"))
        self.corridorcell_widget = QSpinBox()
        self.corridorcell_widget.setValue(1)
        self.corridorcell_widget.setMinimum(1)
        self.corridorcell_widget.setMaximum(config.dungeon.corridor_max_length)
        self.corridorcell_widget.valueChanged.connect(self.corridorcell_changed)
        corridorcell_layout.addWidget(self.corridorcell_widget)
        self.layout.addWidget(self.corridorcell_container)

        self.layout.addWidget(QLabel("Which type?"))
        self.type_widget = QComboBox()
        self.type_widget.addItems([t.value for t in EntityEnum])
        self.type_widget.setCurrentText(EntityEnum.ENEMY.value)
        self.type_widget.currentTextChanged.connect(self.type_changed)
        self.layout.addWidget(self.type_widget)

        self.name_label = QLabel(f"Which {EntityEnum.ENEMY.value}?")
        self.layout.addWidget(self.name_label)
        self.name_widget = QComboBox()
        self.layout.addWidget(self.name_widget)

        self.submit_btn.setText("Remove enemy")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

        self.roomname_changed(self.level.current_room)

    def roomname_changed(self, roomname: str) -> None:
        if roomname in self.level.rooms.keys():
            self.corridorcell_container.hide()
        else:
            self.corridorcell_container.show()
            corridor = self.level.corridors[roomname]
            self.corridorcell_widget.setMaximum(corridor.length)
        self.refresh_names()

    def type_changed(self, t: str) -> None:
        self.name_label.setText(f"Which {t}?")
        self.submit_btn.setText(f"Remove {t}")
        self.refresh_names()

    def corridorcell_changed(self, v: int) -> None:
        self.refresh_names()

    def refresh_names(self):
        t_enum = get_enum_by_value(EntityEnum, self.type_widget.currentText())
        if self.roomname_widget.currentText() in self.level.rooms.keys():
            encounter = self.level.rooms[self.roomname_widget.currentText()].encounter
        else:
            encounter = self.level.corridors[
                self.roomname_widget.currentText()
            ].encounters[self.corridorcell_widget.value()]
        self.name_widget.clear()
        self.name_widget.addItems([x.name for x in encounter.entities[t_enum.value]])

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        cell_index = (
            self.corridorcell_widget.value()
            if room_name in self.level.corridors.keys()
            else -1
        )

        return {
            "self": None,
            "level": self.level,
            "room_name": room_name,
            "cell_index": cell_index,
            "entity_type": self.type_widget.currentText(),
            "entity_name": self.name_widget.currentText(),
        }


class AddAttackDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Add Attack")

        self.layout.addWidget(QLabel("In which room/corridor?"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.addItems(list(self.level.corridors.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.roomname_widget.currentTextChanged.connect(self.roomname_changed)
        self.layout.addWidget(self.roomname_widget)

        self.corridorcell_container = QWidget(parent=self)
        corridorcell_layout = QVBoxLayout(self.corridorcell_container)
        corridorcell_layout.addWidget(QLabel("In which corridor cell?"))
        self.corridorcell_widget = QSpinBox()
        self.corridorcell_widget.setValue(1)
        self.corridorcell_widget.setMinimum(1)
        self.corridorcell_widget.setMaximum(config.dungeon.corridor_max_length)
        corridorcell_layout.addWidget(self.corridorcell_widget)
        self.layout.addWidget(self.corridorcell_container)

        self.enemy_name_label = QLabel(f"Which {EntityEnum.ENEMY.value}?")
        self.layout.addWidget(self.enemy_name_label)
        self.enemy_name_widget = QComboBox()
        self.layout.addWidget(self.enemy_name_widget)

        self.layout.addWidget(QLabel("Name:"))
        self.name_widget = QLineEdit()
        self.layout.addWidget(self.name_widget)

        self.layout.addWidget(QLabel("Description"))
        self.description_widget = QLineEdit()
        self.layout.addWidget(self.description_widget)

        self.layout.addWidget(QLabel("Which attack type?"))
        self.type_widget = QComboBox()
        self.type_widget.addItems(
            [t.value for t in [ActionType.DAMAGE, ActionType.HEAL]]
        )
        self.type_widget.setCurrentText(ActionType.DAMAGE.value)
        self.layout.addWidget(self.type_widget)

        self.startpos_container = QWidget(parent=self)
        startpos_layout = QHBoxLayout(self.startpos_container)
        startpos_layout.addWidget(QLabel("Starting Positions:"))
        for i in range(config.dungeon.max_enemies_per_encounter):
            startpos_layout.addWidget(QCheckBox(f"@{i + 1}"))
        self.layout.addWidget(self.startpos_container)

        self.targetpos_container = QWidget(parent=self)
        targetpos_layout = QHBoxLayout(self.targetpos_container)
        targetpos_layout.addWidget(QLabel("Target Positions:"))
        for i in range(config.dungeon.max_enemies_per_encounter):
            targetpos_layout.addWidget(QCheckBox(f"@{i + 1}"))
        self.layout.addWidget(self.targetpos_container)

        self.dmg_container = QWidget(parent=self)
        dmg_layout = QHBoxLayout(self.dmg_container)
        dmg_layout.addWidget(QLabel("Damage:"))
        self.dmg_widget = QDoubleSpinBox()
        self.dmg_widget.setSingleStep(0.01)
        self.dmg_widget.setValue(config.dungeon.min_base_dmg)
        self.dmg_widget.setMinimum(config.dungeon.min_base_dmg)
        self.dmg_widget.setMaximum(config.dungeon.max_base_dmg)
        dmg_layout.addWidget(self.dmg_widget)
        self.layout.addWidget(self.dmg_container)

        self.accuracy_container = QWidget(parent=self)
        accuracy_layout = QHBoxLayout(self.accuracy_container)
        accuracy_layout.addWidget(QLabel("Accuracy:"))
        self.accuracy_widget = QDoubleSpinBox()
        self.accuracy_widget.setSingleStep(0.01)
        self.accuracy_widget.setValue(0.0)
        self.accuracy_widget.setMinimum(0.0)
        self.accuracy_widget.setMaximum(1.0)
        accuracy_layout.addWidget(self.accuracy_widget)
        self.layout.addWidget(self.accuracy_container)

        self.modifier_container = QWidget(parent=self)
        modifier_layout = QHBoxLayout(self.modifier_container)
        modifier_layout.addWidget(QLabel("Modifier:"))
        self.modifier_widget = QComboBox()
        self.modifier_widget.addItems(["None"] + [x.value for x in ModifierType])
        self.modifier_widget.setCurrentText("None")
        self.modifier_widget.currentTextChanged.connect(self.modifiertype_changed)
        modifier_layout.addWidget(self.modifier_widget)
        self.layout.addWidget(self.modifier_container)

        self.modifierchance_container = QWidget(parent=self)
        modifierchance_layout = QHBoxLayout(self.modifierchance_container)
        modifierchance_layout.addWidget(QLabel("Modifier Chance:"))
        self.modifierchance_widget = QDoubleSpinBox()
        self.modifierchance_widget.setSingleStep(0.01)
        self.modifierchance_widget.setValue(0.0)
        self.modifierchance_widget.setMinimum(0.0)
        self.modifierchance_widget.setMaximum(1.0)
        modifierchance_layout.addWidget(self.modifierchance_widget)
        self.layout.addWidget(self.modifierchance_container)

        self.modifierturns_container = QWidget(parent=self)
        modifierturns_layout = QHBoxLayout(self.modifierturns_container)
        modifierturns_layout.addWidget(QLabel("Modifier Turns:"))
        self.modifierturns_widget = QSpinBox()
        self.modifierturns_widget.setSingleStep(1)
        self.modifierturns_widget.setValue(0)
        self.modifierturns_widget.setMinimum(0)
        self.modifierturns_widget.setMaximum(5)
        modifierturns_layout.addWidget(self.modifierturns_widget)
        self.layout.addWidget(self.modifierturns_container)

        self.modifieramount_container = QWidget(parent=self)
        modifieramount_layout = QHBoxLayout(self.modifieramount_container)
        modifieramount_layout.addWidget(QLabel("Modifier Amount:"))
        self.modifieramount_widget = QDoubleSpinBox()
        self.modifieramount_widget.setSingleStep(0.01)
        self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
        modifieramount_layout.addWidget(self.modifieramount_widget)
        self.layout.addWidget(self.modifieramount_container)

        self.submit_btn.setText("Add Attack")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

        self.roomname_changed(self.level.current_room)
        self.modifiertype_changed("None")

    def get_encounter(self) -> Encounter:
        if self.roomname_widget.currentText() in self.level.rooms.keys():
            encounter = self.level.rooms[self.roomname_widget.currentText()].encounter
        else:
            encounter = self.level.corridors[
                self.roomname_widget.currentText()
            ].encounters[self.corridorcell_widget.value()]
        return encounter

    def roomname_changed(self, roomname: str) -> None:
        if roomname in self.level.rooms.keys():
            self.corridorcell_container.hide()
        else:
            self.corridorcell_container.show()
            corridor = self.level.corridors[roomname]
            self.corridorcell_widget.setMaximum(corridor.length)
        self.refresh_enemy_names()

    def corridorcell_changed(self, v: int) -> None:
        self.refresh_enemy_names()

    def refresh_enemy_names(self) -> None:
        encounter = self.get_encounter()
        self.enemy_name_widget.clear()
        self.enemy_name_widget.addItems([x.name for x in encounter.enemies])

    def modifiertype_changed(self, modifiertype: str) -> None:
        if modifiertype == "None":
            self.modifierchance_container.hide()
            self.modifierturns_container.hide()
            self.modifieramount_container.hide()
        else:
            self.modifierchance_container.show()
            self.modifierturns_container.show()

            m_type = get_enum_by_value(ModifierType, modifiertype)

            if m_type == ModifierType.BLEED or m_type == ModifierType.HEAL:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
            elif m_type == ModifierType.SCARE:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(0.0)
                self.modifieramount_widget.setMinimum(0.0)
                self.modifieramount_widget.setMaximum(1.0)
            else:  # m_type is STUN
                self.modifieramount_container.hide()

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        cell_index = (
            self.corridorcell_widget.value()
            if room_name in self.level.corridors.keys()
            else -1
        )
        starting_positions = "".join(
            [
                "X" if x.isChecked() else "O"
                for x in self.startpos_container.children()[2:]
            ]
        )
        target_positions = "".join(
            [
                "X" if x.isChecked() else "O"
                for x in self.targetpos_container.children()[2:]
            ]
        )
        m_str = self.modifier_widget.currentText()
        if m_str == "None":
            m_str = "no-modifier"
        return {
            "self": None,
            "level": self.level,
            "room_name": room_name,
            "cell_index": cell_index,
            "enemy_name": self.enemy_name_widget.currentText(),
            "name": self.name_widget.text(),
            "description": self.description_widget.text(),
            "attack_type": self.type_widget.currentText(),
            "starting_positions": starting_positions,
            "target_positions": target_positions,
            "base_dmg": self.dmg_widget.value(),
            "accuracy": self.accuracy_widget.value(),
            "modifier_type": m_str,
            "modifier_chance": self.modifierchance_widget.value(),
            "modifier_turns": self.modifierturns_widget.value(),
            "modifier_amount": self.modifieramount_widget.value(),
        }


class UpdateAttackDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Add Attack")

        self.layout.addWidget(QLabel("In which room/corridor?"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.addItems(list(self.level.corridors.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.roomname_widget.currentTextChanged.connect(self.roomname_changed)
        self.layout.addWidget(self.roomname_widget)

        self.corridorcell_container = QWidget(parent=self)
        corridorcell_layout = QVBoxLayout(self.corridorcell_container)
        corridorcell_layout.addWidget(QLabel("In which corridor cell?"))
        self.corridorcell_widget = QSpinBox()
        self.corridorcell_widget.setValue(1)
        self.corridorcell_widget.setMinimum(1)
        self.corridorcell_widget.setMaximum(config.dungeon.corridor_max_length)
        corridorcell_layout.addWidget(self.corridorcell_widget)
        self.layout.addWidget(self.corridorcell_container)

        self.enemy_name_label = QLabel(f"Which {EntityEnum.ENEMY.value}?")
        self.layout.addWidget(self.enemy_name_label)
        self.enemy_name_widget = QComboBox()
        self.enemy_name_widget.currentTextChanged.connect(self.enemy_name_changed)
        self.layout.addWidget(self.enemy_name_widget)

        self.reference_name_label = QLabel(f"Which attack?")
        self.layout.addWidget(self.reference_name_label)
        self.reference_name_widget = QComboBox()
        self.reference_name_widget.currentTextChanged.connect(
            self.reference_name_changed
        )
        self.layout.addWidget(self.reference_name_widget)

        self.layout.addWidget(QLabel("Name:"))
        self.name_widget = QLineEdit()
        self.layout.addWidget(self.name_widget)

        self.layout.addWidget(QLabel("Description"))
        self.description_widget = QLineEdit()
        self.layout.addWidget(self.description_widget)

        self.layout.addWidget(QLabel("Which attack type?"))
        self.type_widget = QComboBox()
        self.type_widget.addItems(
            [t.value for t in [ActionType.DAMAGE, ActionType.HEAL]]
        )
        self.type_widget.setCurrentText(ActionType.DAMAGE.value)
        self.layout.addWidget(self.type_widget)

        self.startpos_container = QWidget(parent=self)
        startpos_layout = QHBoxLayout(self.startpos_container)
        startpos_layout.addWidget(QLabel("Starting Positions:"))
        for i in range(config.dungeon.max_enemies_per_encounter):
            startpos_layout.addWidget(QCheckBox(f"@{i + 1}"))
        self.layout.addWidget(self.startpos_container)

        self.targetpos_container = QWidget(parent=self)
        targetpos_layout = QHBoxLayout(self.targetpos_container)
        targetpos_layout.addWidget(QLabel("Target Positions:"))
        for i in range(config.dungeon.max_enemies_per_encounter):
            targetpos_layout.addWidget(QCheckBox(f"@{i + 1}"))
        self.layout.addWidget(self.targetpos_container)

        self.dmg_container = QWidget(parent=self)
        dmg_layout = QHBoxLayout(self.dmg_container)
        dmg_layout.addWidget(QLabel("Base Damage:"))
        self.dmg_widget = QDoubleSpinBox()
        self.dmg_widget.setSingleStep(0.01)
        self.dmg_widget.setValue(config.dungeon.min_base_dmg)
        self.dmg_widget.setMinimum(config.dungeon.min_base_dmg)
        self.dmg_widget.setMaximum(config.dungeon.max_base_dmg)
        dmg_layout.addWidget(self.dmg_widget)
        self.layout.addWidget(self.dmg_container)

        self.accuracy_container = QWidget(parent=self)
        accuracy_layout = QHBoxLayout(self.accuracy_container)
        accuracy_layout.addWidget(QLabel("Accuracy:"))
        self.accuracy_widget = QDoubleSpinBox()
        self.accuracy_widget.setSingleStep(0.01)
        self.accuracy_widget.setValue(0.0)
        self.accuracy_widget.setMinimum(0.0)
        self.accuracy_widget.setMaximum(1.0)
        accuracy_layout.addWidget(self.accuracy_widget)
        self.layout.addWidget(self.accuracy_container)

        self.modifier_container = QWidget(parent=self)
        modifier_layout = QHBoxLayout(self.modifier_container)
        modifier_layout.addWidget(QLabel("Modifier:"))
        self.modifier_widget = QComboBox()
        self.modifier_widget.addItems(["None"] + [x.value for x in ModifierType])
        self.modifier_widget.setCurrentText("None")
        self.modifier_widget.currentTextChanged.connect(self.modifiertype_changed)
        modifier_layout.addWidget(self.modifier_widget)
        self.layout.addWidget(self.modifier_container)

        self.modifierchance_container = QWidget(parent=self)
        modifierchance_layout = QHBoxLayout(self.modifierchance_container)
        modifierchance_layout.addWidget(QLabel("Modifier Chance:"))
        self.modifierchance_widget = QDoubleSpinBox()
        self.modifierchance_widget.setSingleStep(0.01)
        self.modifierchance_widget.setValue(0.0)
        self.modifierchance_widget.setMinimum(0.0)
        self.modifierchance_widget.setMaximum(1.0)
        modifierchance_layout.addWidget(self.modifierchance_widget)
        self.layout.addWidget(self.modifierchance_container)

        self.modifierturns_container = QWidget(parent=self)
        modifierturns_layout = QHBoxLayout(self.modifierturns_container)
        modifierturns_layout.addWidget(QLabel("Modifier Turns:"))
        self.modifierturns_widget = QSpinBox()
        self.modifierturns_widget.setSingleStep(1)
        self.modifierturns_widget.setValue(0)
        self.modifierturns_widget.setMinimum(0)
        self.modifierturns_widget.setMaximum(5)
        modifierturns_layout.addWidget(self.modifierturns_widget)
        self.layout.addWidget(self.modifierturns_container)

        self.modifieramount_container = QWidget(parent=self)
        modifieramount_layout = QHBoxLayout(self.modifieramount_container)
        modifieramount_layout.addWidget(QLabel("Modifier Amount:"))
        self.modifieramount_widget = QDoubleSpinBox()
        self.modifieramount_widget.setSingleStep(0.01)
        self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
        self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
        modifieramount_layout.addWidget(self.modifieramount_widget)
        self.layout.addWidget(self.modifieramount_container)

        self.submit_btn.setText("Update Attack")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

        self.roomname_changed(self.level.current_room)
        self.modifiertype_changed("None")

    def get_encounter(self) -> Encounter:
        if self.roomname_widget.currentText() in self.level.rooms.keys():
            encounter = self.level.rooms[self.roomname_widget.currentText()].encounter
        else:
            encounter = self.level.corridors[
                self.roomname_widget.currentText()
            ].encounters[self.corridorcell_widget.value()]
        return encounter

    def roomname_changed(self, roomname: str) -> None:
        if roomname in self.level.rooms.keys():
            self.corridorcell_container.hide()
        else:
            self.corridorcell_container.show()
            corridor = self.level.corridors[roomname]
            self.corridorcell_widget.setMaximum(corridor.length)
        self.refresh_enemy_names()

    def corridorcell_changed(self, v: int) -> None:
        self.refresh_enemy_names()

    def refresh_enemy_names(self) -> None:
        encounter = self.get_encounter()
        self.enemy_name_widget.clear()
        self.enemy_name_widget.addItems([x.name for x in encounter.enemies])

    def enemy_name_changed(self, enemy_name: str) -> None:
        self.refresh_names()

    def refresh_names(self):
        encounter = self.get_encounter()
        enemy = encounter.get_entity_by_name(
            entity_type=EntityEnum.ENEMY,
            entity_name=self.enemy_name_widget.currentText(),
        )
        self.name_widget.clear()
        if enemy is not None:
            self.reference_name_widget.addItems([x.name for x in enemy.attacks])

    def reference_name_changed(self, reference_name: str) -> None:
        encounter = self.get_encounter()
        enemy: Enemy = encounter.get_entity_by_name(
            entity_type=EntityEnum.ENEMY,
            entity_name=self.enemy_name_widget.currentText(),
        )
        if enemy is not None:
            attack = enemy.attacks[
                [x.name for x in enemy.attacks].index(reference_name)
            ]
            self.name_widget.setText(attack.name)
            self.description_widget.setText(attack.description)
            self.type_widget.setCurrentText(attack.type)
            for chkbox, c in zip(
                self.startpos_container.children()[2:], attack.starting_positions
            ):
                chkbox.setChecked(c == "X")
            for chkbox, c in zip(
                self.targetpos_container.children()[2:], attack.target_positions
            ):
                chkbox.setChecked(c == "X")
            self.dmg_widget.setValue(attack.base_dmg)
            self.accuracy_widget.setValue(attack.accuracy)
            if attack.modifier is not None:
                self.modifier_widget.setCurrentText(attack.modifier.type)
                self.modifierchance_widget.setValue(attack.modifier.chance)
                self.modifierturns_widget.setValue(attack.modifier.turns)
                self.modifieramount_widget.setValue(attack.modifier.amount)
            else:
                self.modifier_widget.setCurrentText("None")
        else:
            self.name_widget.setText("")
            self.description_widget.setText("")
            self.type_widget.setCurrentText(ActionType.DAMAGE.value)
            for chkbox, c in zip(
                self.startpos_container.children()[2:], attack.starting_positions
            ):
                chkbox.setChecked(False)
            for chkbox, c in zip(
                self.targetpos_container.children()[2:], attack.target_positions
            ):
                chkbox.setChecked(False)
            self.dmg_widget.setValue(config.dungeon.min_base_dmg)
            self.accuracy_widget.setValue(0.0)
            self.modifier_widget.setCurrentText("None")
            self.modifierchance_widget.setValue(0.0)
            self.modifierturns_widget.setValue(0.0)
            self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)

    def modifiertype_changed(self, modifiertype: str) -> None:
        if modifiertype == "None":
            self.modifierchance_container.hide()
            self.modifierturns_container.hide()
            self.modifieramount_container.hide()
        else:
            self.modifierchance_container.show()
            self.modifierturns_container.show()

            m_type = get_enum_by_value(ModifierType, modifiertype)

            if m_type == ModifierType.BLEED or m_type == ModifierType.HEAL:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMinimum(config.dungeon.min_base_dmg)
                self.modifieramount_widget.setMaximum(config.dungeon.max_base_dmg)
            elif m_type == ModifierType.SCARE:
                self.modifieramount_container.show()
                self.modifieramount_widget.setValue(0.0)
                self.modifieramount_widget.setMinimum(0.0)
                self.modifieramount_widget.setMaximum(1.0)
            else:  # m_type is STUN
                self.modifieramount_container.hide()

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        cell_index = (
            self.corridorcell_widget.value()
            if room_name in self.level.corridors.keys()
            else -1
        )
        starting_positions = "".join(
            [
                "X" if x.isChecked() else "O"
                for x in self.startpos_container.children()[2:]
            ]
        )
        target_positions = "".join(
            [
                "X" if x.isChecked() else "O"
                for x in self.targetpos_container.children()[2:]
            ]
        )
        m_str = self.modifier_widget.currentText()
        if m_str == "None":
            m_str = "no-modifier"
        return {
            "self": None,
            "level": self.level,
            "room_name": room_name,
            "cell_index": cell_index,
            "enemy_name": self.enemy_name_widget.currentText(),
            "reference_name": self.reference_name_widget.currentText(),
            "name": self.name_widget.text(),
            "description": self.description_widget.text(),
            "attack_type": self.type_widget.currentText(),
            "starting_positions": starting_positions,
            "target_positions": target_positions,
            "base_dmg": self.dmg_widget.value(),
            "accuracy": self.accuracy_widget.value(),
            "modifier_type": m_str,
            "modifier_chance": self.modifierchance_widget.value(),
            "modifier_turns": self.modifierturns_widget.value(),
            "modifier_amount": self.modifieramount_widget.value(),
        }


class RemoveAttackDialog(UserModeDialog):
    def __init__(self, level, func, parent=None):
        super().__init__(level, func, parent)
        self.setWindowTitle("Remove Attack")

        self.layout.addWidget(QLabel("In which room/corridor?"))
        self.roomname_widget = QComboBox()
        self.roomname_widget.addItems(list(self.level.rooms.keys()))
        self.roomname_widget.addItems(list(self.level.corridors.keys()))
        self.roomname_widget.setCurrentText(self.level.current_room)
        self.roomname_widget.currentTextChanged.connect(self.roomname_changed)
        self.layout.addWidget(self.roomname_widget)

        self.corridorcell_container = QWidget(parent=self)
        corridorcell_layout = QVBoxLayout(self.corridorcell_container)
        corridorcell_layout.addWidget(QLabel("In which corridor cell?"))
        self.corridorcell_widget = QSpinBox()
        self.corridorcell_widget.setValue(1)
        self.corridorcell_widget.setMinimum(1)
        self.corridorcell_widget.setMaximum(config.dungeon.corridor_max_length)
        self.corridorcell_widget.valueChanged.connect(self.corridorcell_changed)
        corridorcell_layout.addWidget(self.corridorcell_widget)
        self.layout.addWidget(self.corridorcell_container)

        self.enemy_name_label = QLabel(f"Which {EntityEnum.ENEMY.value}?")
        self.layout.addWidget(self.enemy_name_label)
        self.enemy_name_widget = QComboBox()
        self.enemy_name_widget.currentTextChanged.connect(self.enemy_name_changed)
        self.layout.addWidget(self.enemy_name_widget)

        self.name_label = QLabel(f"Which attack?")
        self.layout.addWidget(self.name_label)
        self.name_widget = QComboBox()
        self.layout.addWidget(self.name_widget)

        self.submit_btn.setText("Remove Attack")
        self.layout.addWidget(self.submit_btn)
        self.layout.addWidget(self.pbar)

        self.roomname_changed(self.level.current_room)

    def get_encounter(self) -> Encounter:
        if self.roomname_widget.currentText() in self.level.rooms.keys():
            encounter = self.level.rooms[self.roomname_widget.currentText()].encounter
        else:
            encounter = self.level.corridors[
                self.roomname_widget.currentText()
            ].encounters[self.corridorcell_widget.value()]
        return encounter

    def roomname_changed(self, roomname: str) -> None:
        if roomname in self.level.rooms.keys():
            self.corridorcell_container.hide()
        else:
            self.corridorcell_container.show()
            corridor = self.level.corridors[roomname]
            self.corridorcell_widget.setMaximum(corridor.length)
        self.refresh_enemy_names()

    def corridorcell_changed(self, v: int) -> None:
        self.refresh_enemy_names()

    def refresh_enemy_names(self) -> None:
        encounter = self.get_encounter()
        self.enemy_name_widget.clear()
        self.enemy_name_widget.addItems([x.name for x in encounter.enemies])

    def enemy_name_changed(self, enemy_name: str) -> None:
        self.refresh_names()

    def refresh_names(self):
        encounter = self.get_encounter()
        enemy = encounter.get_entity_by_name(
            entity_type=EntityEnum.ENEMY,
            entity_name=self.enemy_name_widget.currentText(),
        )
        self.name_widget.clear()
        if enemy is not None:
            self.name_widget.addItems([x.name for x in enemy.attacks])

    def get_kwargs(self) -> Dict[str, Any]:
        room_name = self.roomname_widget.currentText()
        cell_index = (
            self.corridorcell_widget.value()
            if room_name in self.level.corridors.keys()
            else -1
        )

        return {
            "self": None,
            "level": self.level,
            "room_name": room_name,
            "cell_index": cell_index,
            "entity_name": self.enemy_name_widget.currentText(),
            "name": self.name_widget.currentText(),
        }


function_to_dialog = {
    "add_room": AddRoomDialog,
    "remove_room": RemoveRoomDialog,
    "update_room": UpdateRoomDialog,
    "add_corridor": AddCorridorDialog,
    "update_corridor": UpdateCorridorDialog,
    "remove_corridor": RemoveCorridorDialog,
    "add_attack": AddAttackDialog,
    "update_attack": UpdateAttackDialog,
    "remove_attack": RemoveAttackDialog,
}


def check_available_action(level: Level, dialogclass) -> bool:
    if dialogclass == AddRoomDialog:
        return True  # create room is always possible
    elif dialogclass == RemoveRoomDialog or dialogclass == UpdateRoomDialog:
        return len(level.rooms.keys()) > 0
    elif dialogclass == AddCorridorDialog:
        # minimum number of rooms to add a new corridor is 4

        return len(level.rooms.keys()) > 3
    elif dialogclass == UpdateCorridorDialog or dialogclass == RemoveCorridorDialog:
        return len(level.corridors.keys()) > 0
    elif dialogclass == AddEntityDialog:
        return len(level.rooms.keys()) > 0
    elif dialogclass == UpdateEntityDialog or dialogclass == RemoveEntityDialog:
        has_enemy, has_treasure, has_trap = False, False, False
        for room in level.rooms.values():
            has_enemy |= len(room.encounter.enemies) > 0
            has_treasure |= len(room.encounter.treasures) > 0
        for corridor in level.corridors.values():
            for encounter in corridor.encounters:
                has_enemy |= len(encounter.enemies) > 0
                has_treasure |= len(encounter.treasures) > 0
                has_trap |= len(encounter.traps) > 0
        return has_enemy or has_trap or has_treasure
    elif dialogclass == AddAttackDialog:
        has_enemy = False
        for room in level.rooms.values():
            has_enemy |= len(room.encounter.enemies) > 0
        for corridor in level.corridors.values():
            for encounter in corridor.encounters:
                has_enemy |= len(encounter.enemies) > 0
        return has_enemy
    elif dialogclass == UpdateAttackDialog or dialogclass == RemoveAttackDialog:
        has_attack = False
        for room in level.rooms.values():
            for enemy in room.encounter.enemies:
                has_attack |= len(enemy.attacks) > 0
        for corridor in level.corridors.values():
            for encounter in corridor.encounters:
                for enemy in encounter.enemies:
                    has_attack |= len(enemy.attacks) > 0
        return has_attack
    else:
        raise NotImplementedError(f"No validity check for {dialogclass}")
