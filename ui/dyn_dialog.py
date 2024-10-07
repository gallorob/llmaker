import logging
import os

from PyQt6.QtCore import pyqtSlot, QThread, QSize, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, \
	QDoubleSpinBox, QMessageBox, QProgressBar, QGridLayout, QDialogButtonBox, QScrollArea, QWidget
from gptfunctionutil import LibCommand

from configs import config
from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.room import Room
from sd_backend import generate_room, generate_corridor, generate_entity
from ui.input_process import DebugInputProcessor
from utils import compute_level_diffs


class EnemyPreviewDialog(QDialog):
	def __init__(self, enemy: Enemy, parent=None):
		super().__init__(parent)
		
		self.setWindowTitle(f"Details for {enemy.name}")
		self.setWindowIcon(QIcon('assets/llmaker_logo.png'))
		self.setMinimumSize(QSize(400, 300))
		
		layout = QGridLayout(self)
		
		pixmap = QPixmap(os.path.join(config.entity.save_dir, enemy.sprite))
		
		sprite_label = QLabel()
		sprite_label.setPixmap(pixmap.scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatio))
		
		# Enemy details
		details_layout = QVBoxLayout()
		
		details_layout.addWidget(QLabel(f"<b>Name</b>: {enemy.name}"))
		details_layout.addWidget(QLabel(f"<b>Description</b>: {enemy.description}"))
		details_layout.addWidget(QLabel(f"<b>Species</b>: {enemy.species}"))
		details_layout.addWidget(QLabel(f"<b>HP</b>: {enemy.hp}"))
		details_layout.addWidget(QLabel(f"<b>Dodge</b>: {enemy.dodge}"))
		details_layout.addWidget(QLabel(f"<b>Prot</b>: {enemy.prot}"))
		details_layout.addWidget(QLabel(f"<b>Spd</b>: {enemy.spd}"))
		
		# Layout for Attacks
		attacks_layout = QVBoxLayout()
		attacks_layout.addWidget(QLabel("<b>Attacks:</b>"))  # Section title
		
		# Create a scrollable area for attacks in case there are too many
		attacks_scroll_area = QScrollArea(parent=self)
		attacks_scroll_widget = QWidget(parent=self)
		attacks_scroll_widget.setProperty('attacks_grid', 'yes')
		attacks_grid_layout = QGridLayout(attacks_scroll_widget)
		
		# Add headers for the grid
		attacks_grid_layout.addWidget(QLabel("<b>Name</b>"), 0, 0)
		attacks_grid_layout.addWidget(QLabel("<b>Description</b>"), 0, 1)
		attacks_grid_layout.addWidget(QLabel("<b>From</b>"), 0, 2)
		attacks_grid_layout.addWidget(QLabel("<b>To</b>"), 0, 3)
		attacks_grid_layout.addWidget(QLabel("<b>Base Damage</b>"), 0, 4)
		
		# Populate the grid with attacks
		for row, attack in enumerate(enemy.attacks, start=1):
			attacks_grid_layout.addWidget(QLabel(attack.name), row, 0)
			attacks_grid_layout.addWidget(QLabel(attack.description), row, 1)
			attacks_grid_layout.addWidget(QLabel(attack.starting_positions), row, 2)
			attacks_grid_layout.addWidget(QLabel(attack.target_positions), row, 3)
			attacks_grid_layout.addWidget(QLabel(str(attack.base_dmg)), row, 4)
		
		# Set up the scroll area for the attacks list
		attacks_scroll_area.setWidget(attacks_scroll_widget)
		attacks_scroll_area.setWidgetResizable(True)
		
		# Add the scroll area to the layout
		attacks_layout.addWidget(attacks_scroll_area)
		
		# Add a button box for OK button
		button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
		button_box.accepted.connect(self.accept)
		
		# Layout the dialog
		layout.addWidget(sprite_label, 0, 0)
		layout.addLayout(details_layout, 0, 1)
		layout.addLayout(attacks_layout, 1, 0, 1, 2)
		layout.addWidget(button_box, 2, 0, 1, 2)


class DebugFunctionsDialog(QDialog):
	def __init__(self, level, func, parent=None):
		super().__init__(parent)
		self.level = level
		self.func: LibCommand = func
		self.initUI()
		
		self.setWindowTitle(self.func.internal_name)
		self.setWindowIcon(QIcon('assets/llmaker_logo.png'))
	
	def initUI(self):
		layout = QVBoxLayout()
		self.inputs = {}
		
		for param_name, param_properties in self.func.function_schema['parameters']['properties'].items():
			
			param_desc = param_properties['description']
			param_type = param_properties['type']
			layout.addWidget(QLabel(f'{param_name}:\n{param_desc}'))
			
			if param_type == 'integer':
				widget = QSpinBox()
				widget.setMinimum(-1)
				widget.setMaximum(100)
			elif param_type == 'number':
				widget = QDoubleSpinBox()
				widget.setMinimum(-999999.0)
				widget.setMaximum(999999.0)
			elif param_type == 'string':
				widget = QLineEdit()
			else:
				widget = QLineEdit()
			
			layout.addWidget(widget)
			self.inputs[param_name] = widget
		
		# Add submit button
		submit_btn = QPushButton("Submit")
		submit_btn.clicked.connect(self.submit)
		layout.addWidget(submit_btn)
		
		self.pbar = QProgressBar(self)
		self.pbar.setRange(0, 100)
		self.pbar.setHidden(True)
		
		layout.addWidget(self.pbar)
		
		self.setLayout(layout)
	
	@pyqtSlot(int)
	def update_progress(self, progress):
		logging.getLogger().debug(f'update_progress Task progress: {progress}')
		self.pbar.setValue(progress)
	
	@pyqtSlot(str)
	def task_finished(self, result):
		logging.getLogger().debug(f'task_finished Edit finished')
		self.pbar.reset()
		self.pbar.setHidden(True)
		button_pressed = QMessageBox.information(self, "Output", f"{result}")
		if button_pressed == QMessageBox.StandardButton.Ok:
			self.close()
	
	def submit(self):
		kwargs = {'self': None, 'level': self.level}
		for param_name, widget in self.inputs.items():
			if isinstance(widget, QLineEdit):
				kwargs[param_name] = widget.text()
			elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
				kwargs[param_name] = widget.value()
		
		self.worker = DebugInputProcessor(kwargs,
		                                  self.func,
		                                  self)
		self.thread = QThread()
		
		self.worker.moveToThread(self.thread)
		
		self.thread.started.connect(self.worker.run)
		self.worker.finished.connect(self.task_finished)
		self.worker.progress.connect(self.update_progress)
		self.worker.finished.connect(self.thread.quit)
		self.worker.finished.connect(self.worker.deleteLater)
		self.thread.finished.connect(self.thread.deleteLater)
		
		self.pbar.setHidden(False)
		self.pbar.reset()
		
		self.thread.start()
