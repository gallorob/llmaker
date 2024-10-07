import logging

from PyQt6.QtCore import pyqtSlot, QThread
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, \
	QDoubleSpinBox, QMessageBox, QProgressBar
from gptfunctionutil import LibCommand

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.room import Room
from sd_backend import generate_room, generate_corridor, generate_entity
from ui.input_process import DebugInputProcessor
from utils import compute_level_diffs


class DynamicDialog(QDialog):
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
