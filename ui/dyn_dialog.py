import logging

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, \
	QDoubleSpinBox, QMessageBox
from gptfunctionutil import LibCommand

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.room import Room
from sd_backend import generate_room, generate_corridor, generate_entity
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
		
		self.setLayout(layout)
	
	def submit(self):
		# TODO: Would be nice to have it wait while the program generates sprites etc before the dialog closes...
		#  Use the thread thingy and a progress bar like in the main window
		kwargs = {'self': None, 'level': self.level}
		for param_name, widget in self.inputs.items():
			if isinstance(widget, QLineEdit):
				kwargs[param_name] = widget.text()
			elif isinstance(widget, QSpinBox) or isinstance(widget, QDoubleSpinBox):
				kwargs[param_name] = widget.value()
		
		try:
			result = self.func.command(**kwargs)
			button_pressed = QMessageBox.information(self, "Output", f"{result}")
			if button_pressed == QMessageBox.StandardButton.Ok:
				self.close()
		except Exception as e:
			QMessageBox.critical(self, "Error", str(e))
			raise e
		
		try:
			to_process, additional_data = compute_level_diffs(level=self.level)

			for i, obj in enumerate(to_process):
				if isinstance(obj, Room):
					logging.info(f'Room {obj.name} has no sprite; generating...')
					obj.sprite = generate_room(room_name=obj.name,
					                           room_description=obj.description)
				elif isinstance(obj, Corridor):
					logging.info(f'Room {obj.name} has no sprite; generating...')
					obj_data = additional_data[i]
					obj.sprites = generate_corridor(room_names=[obj.room_from, obj.room_to],
					                               corridor_length=obj.length + 2,
					                               **obj_data,
					                                corridor_sprites=obj.sprites)
				elif isinstance(obj, Entity):
					obj_data = additional_data[i]
					logging.info(f'Entity {obj.name} has no sprite; generating...')
					obj.sprite = generate_entity(entity_name=obj.name,
					                             entity_description=obj.description,
					                             **obj_data)
				else:
					raise ValueError(f'Unsupported object type: {type(obj)}')
		except Exception as e:
			QMessageBox.critical(self, "Error", str(e))
			raise e
