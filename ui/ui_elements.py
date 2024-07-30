import logging
import os
import time
from enum import Enum, auto
from functools import partial
from typing import Any, List, Tuple, Union

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QIcon, QPixmap
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import QErrorMessage, QFileDialog, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, \
	QGraphicsView, \
	QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar, QPushButton, QSplashScreen, \
	QVBoxLayout, QWidget, QSizePolicy, QScrollArea

from configs import config
from level import Corridor, DIRECTIONS, Entity, Level, OPPOSITE_DIRECTIONS, Room
from llm_backend import chat_llm
from sd_backend import generate_corridor, generate_entity, generate_room
from utils import basic_corridor_description, basic_room_description, rich_entity_description


# TODO: Reformat stylesheets (avoid hardcoded colors in components here)
# TODO: Set message property (user/assistant) instead of assuming alternating roles in conversation


def get_splash_screen():
	pixmap = QPixmap('assets/llmaker_splash.png')
	splash = QSplashScreen(pixmap)
	return splash


class InputProcessor(QObject):
	finished = pyqtSignal()
	progress = pyqtSignal(int)
	result = pyqtSignal(str)
	
	def __init__(self,
	             level: Level,
	             user_input: str,
	             conversation_history: str):
		super(InputProcessor, self).__init__()
		self.level = level
		self.user_input = user_input
		self.conversation_history = conversation_history
	
	def run(self) -> str:
		self.progress_n = 0
		ai_response = chat_llm(user_message=self.user_input,
		                       conversation_history=self.conversation_history,
		                       level=self.level)
		self.result.emit(ai_response)
		
		# compute the % of each progress
		to_process: List[Union[Room, Corridor, Entity]] = []
		additional_data: List[Any] = []
		
		for room_name in self.level.rooms.keys():
			room = self.level.rooms[room_name]
			if room.sprite is None:
				to_process.append(room)
				additional_data.append(None)
			for entity_type in room.encounter.entities:
				for entity in room.encounter.entities[entity_type]:
					if entity.sprite is None:
						to_process.append(entity)
						additional_data.append(
							{'entity_type': entity_type, 'room_name': room.name, 'room_description': room.description})
		for corridor in self.level.corridors:
			if corridor.sprite is None:
				to_process.append(corridor)
				room_from, room_to = self.level.rooms[corridor.room_from], self.level.rooms[corridor.room_to]
				additional_data.append({'room_descriptions': [room_from.description, room_to.description]})
			for i, encounter in enumerate(corridor.encounters):
				for entity_type in encounter.entities:
					for entity in encounter.entities[entity_type]:
						if entity.sprite is None:
							to_process.append(entity)
							room_from = self.level.rooms[corridor.room_from]
							additional_data.append(
								{'entity_type': entity_type, 'room_name': room_from.name,
								 'room_description': room_from.description})
		
		progress_delta = int((1 / (1 + len(to_process))) * 100)
		
		self.progress_n += progress_delta
		self.progress.emit(self.progress_n)
		
		for i, obj in enumerate(to_process):
			if isinstance(obj, Room):
				logging.info(f'Room {obj.name} has no sprite; generating...')
				obj.sprite = generate_room(room_name=obj.name,
				                           room_description=obj.description)
			elif isinstance(obj, Corridor):
				logging.info(f'Room {obj.name} has no sprite; generating...')
				obj_data = additional_data[i]
				obj.sprite = generate_corridor(room_names=[obj.room_from, obj.room_to],
				                               corridor_length=obj.length + 2,
				                               **obj_data)
			elif isinstance(obj, Entity):
				obj_data = additional_data[i]
				logging.info(f'Entity {obj.name} has no sprite; generating...')
				obj.sprite = generate_entity(entity_name=obj.name,
				                             entity_description=obj.description,
				                             **obj_data)
			else:
				raise ValueError(f'Unsupported object type: {type(obj)}')
			self.progress_n += progress_delta
			self.progress.emit(self.progress_n)
		
		time.sleep(0.5)
		self.finished.emit()


class RoomPreviewWidget(QWidget):
	def __init__(self, parent, level: Level):
		super(RoomPreviewWidget, self).__init__(parent)
		
		self.level = level
		
		self.scene = QGraphicsScene(self)
		self.view = QGraphicsView(self.scene)
		
		self.view_layout = QVBoxLayout(self)
		self.view_layout.addWidget(self.view)
	
	def paintEvent(self, a0):
		self.scene.clear()
		self.show_room_preview()
	
	def show_room_preview(self):
		self.scene.setBackgroundBrush(
			QBrush(QColor('#1e1d23' if self.parent().parent().parent().theme == 'DARK' else '#ececec')))
		
		if self.level.current_room != '':
			if self.level.current_room in self.level.rooms:
				room = self.level.rooms[self.level.current_room]
			else:
				room = self.level.get_corridor(*self.level.current_room.split('-'))
			
			# background_image = QPixmap.fromImage(ImageQt(room.sprite.convert("RGBA")))
			background_image = QPixmap(room.sprite)
			self.scene.setSceneRect(0, 0, background_image.width(), background_image.height())
			item = self.scene.addPixmap(background_image)
			item.setPos(0, 0)
			self.view.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
			
			w, h = self.scene.width(), self.scene.height()
			
			def __draw_entities(entities: List[Entity], x_offset, y_offset, scaled_entity_width) -> None:
				for i in range(config.dungeon.max_enemies_per_encounter):
					if i < len(entities):
						entity = entities[i]
						# entity_sprite = QPixmap.fromImage(ImageQt(entity.sprite.convert("RGBA")))
						entity_sprite = QPixmap(entity.sprite)
						entity_rect = QGraphicsPixmapItem(entity_sprite)
						entity_rect.setScale(config.ui.entity_scale)
						entity_rect.setToolTip(rich_entity_description(entity=entity))
						entity_rect.setPos(x_offset + scaled_entity_width * i,
						                   y_offset - (entity_sprite.height() * entity_rect.scale()))
						self.scene.addItem(entity_rect)
			
			if isinstance(room, Room):
				scaled_entity_width = config.entity.width * config.ui.entity_scale
				y_offset = 5 * h / 6
				x_offset = w / 2 - scaled_entity_width / 2
				
				__draw_entities(room.encounter.entities['treasure'], x_offset, y_offset, scaled_entity_width)
				
				enemies = room.encounter.entities['enemy']
				if len(enemies) > 1:
					x_offset -= (scaled_entity_width * (len(enemies) - 1)) / 2
				__draw_entities(enemies, x_offset, y_offset, scaled_entity_width)
			else:
				scaled_entity_width = config.entity.width * config.ui.entity_scale
				y_offset = 0.9 * h
				for i, encounter in enumerate(room.encounters):
					x_offset = ((i + 1) * w / (room.length + 2)) + w / (room.length + 2) / 2 - (scaled_entity_width / 2)
					
					__draw_entities(encounter.entities['treasure'], x_offset, y_offset, scaled_entity_width)
					__draw_entities(encounter.entities['trap'], x_offset, y_offset, scaled_entity_width)
					
					enemies = encounter.entities['enemy']
					if len(enemies) > 1:
						x_offset -= (scaled_entity_width * (len(enemies) - 1)) / 2
					__draw_entities(enemies, x_offset, y_offset, scaled_entity_width)


class ConversationWidget(QWidget):
	def __init__(self, parent):
		super(ConversationWidget, self).__init__(parent)
		
		self.scroll_area = QScrollArea(self)
		self.scroll_area.setWidgetResizable(True)
		
		self.central_widget = QWidget(self)
		self.central_widget.setProperty('conversation', 'yes')
		self.scroll_area.setWidget(self.central_widget)
		
		self.central_layout = QVBoxLayout(self.central_widget)
		self.central_layout.setSpacing(0)
		self.central_layout.setContentsMargins(0, 0, 0, 0)
		
		# Hacky way to avoid resizing messages when they are too few
		# Praise https://stackoverflow.com/questions/63438039/qt-dont-stretch-widgets-in-qvboxlayout
		self.central_layout.addStretch()
		
		main_layout = QVBoxLayout(self)
		main_layout.addWidget(self.scroll_area)
		self.setLayout(main_layout)
		
		self.messages: List[QLabel] = []
	
	def reset(self):
		for message in self.messages:
			self.central_layout.removeWidget(message)
			message.deleteLater()
		self.messages.clear()
		self.update()
	
	def load_conversation(self, conversation):
		for i, line in enumerate(conversation.split('\n')):
			line = line.replace('You: ', '').replace('AI: ', '')
			self.add_message(line)
	
	def add_message(self, message):
		new_message = QLabel(parent=self.central_widget, text=message)
		new_message.setProperty('messageType', 'me' if len(self.messages) % 2 == 0 else 'them')
		
		new_message.setWordWrap(True)
		new_message.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
		
		# Set maximum width to prevent horizontal stretching
		new_message.setMaximumWidth(int(self.width()))
		
		self.messages.append(new_message)
		self.central_layout.addWidget(new_message)
		
		self.update()
	
	def resizeEvent(self, event):
		# Adjust maximum width of all messages on resize
		for message in self.messages:
			message.setMaximumWidth(self.width())
		super().resizeEvent(event)
	
	def get_conversation(self):
		return '\n'.join([message.text() for message in self.messages])


class MapPreviewWidget(QWidget):
	def __init__(self, parent, level: Level):
		super(MapPreviewWidget, self).__init__(parent)
		
		self.level = level
		
		self.scene = QGraphicsScene(self)
		self.view = QGraphicsView(self.scene)
		
		self.view_layout = QVBoxLayout(self)
		self.view_layout.addWidget(self.view)
		
		self.drawn_rooms = []
		
		self.corridor_draw_size = self.rect().height() * config.ui.minimap_corridor_scale
		self.room_draw_size = self.rect().height() * config.ui.minimap_room_scale
	
	def paintEvent(self, a0):
		self.show_map_preview()
	
	def get_rects(self, room: Union[Room, Corridor], offset_x, offset_y, direction, selected=False) -> Tuple[
		List[QGraphicsRectItem], int, int]:
		def __update_offsets(offset_x, offset_y, direction, draw_size) -> Tuple[int, int]:
			if direction == 'UP':
				offset_y -= draw_size
			elif direction == 'DOWN':
				offset_y += draw_size
			elif direction == 'LEFT':
				offset_x -= draw_size
			elif direction == 'RIGHT':
				offset_x += draw_size
			else:
				pass
			return offset_x, offset_y
		
		rects = []
		
		if isinstance(room, Room):
			draw_offset_x, draw_offset_y = offset_x, offset_y
			rect = QGraphicsRectItem(0, 0, self.room_draw_size, self.room_draw_size)
			rect.setBrush(QBrush(QColor(config.ui.selected_color if selected else config.ui.unselected_color)))
			rect.setToolTip(basic_room_description(room))
			draw_offset_x -= self.room_draw_size / 2
			draw_offset_y -= self.room_draw_size / 2
			rect.setPos(draw_offset_x, draw_offset_y)
			rect.mousePressEvent = partial(self.parent().parent().parent().on_room_press, room.name)
			rects.append(rect)
			self.drawn_rooms.append(room.name)
			
			for direction in DIRECTIONS:
				other_room_name = self.level.level_geometry[room.name][direction]
				if other_room_name != '':
					corridor = self.level.get_corridor(room_from_name=room.name, room_to_name=other_room_name)
					if corridor is not None:
						if f'{corridor.room_from}_{corridor.room_to}' not in self.drawn_rooms:
							next_offset_x, next_offset_y = __update_offsets(offset_x, offset_y, direction,
							                                                self.room_draw_size)
							other_rects, _, _ = self.get_rects(corridor, next_offset_x, next_offset_y, direction)
							rects.extend(other_rects)
		
		else:
			corridor_offset = (self.room_draw_size - self.corridor_draw_size) / 2
			if direction != '':
				draw_direction = direction
				draw_offset_x, draw_offset_y = __update_offsets(offset_x, offset_y, OPPOSITE_DIRECTIONS[direction],
				                                                corridor_offset)
			else:
				draw_offset_x, draw_offset_y = offset_x, offset_y
				corridor_draw_length_offset = self.corridor_draw_size * room.length / 2
				if self.level.level_geometry[room.room_from]['UP'] == room.room_to or \
						self.level.level_geometry[room.room_from]['DOWN'] == room.room_to:
					draw_direction = 'DOWN'
					draw_offset_y -= corridor_draw_length_offset
				else:
					draw_direction = 'RIGHT'
					draw_offset_x -= corridor_draw_length_offset
			
			draw_offset_x -= self.corridor_draw_size / 2
			draw_offset_y -= self.corridor_draw_size / 2
			for i in range(room.length):
				rect = QGraphicsRectItem(0, 0, self.corridor_draw_size, self.corridor_draw_size)
				rect.setBrush(QBrush(QColor(config.ui.selected_color if selected else config.ui.unselected_color)))
				rect.setToolTip(basic_corridor_description(room))
				rect.setPos(draw_offset_x, draw_offset_y)
				draw_offset_x, draw_offset_y = __update_offsets(draw_offset_x, draw_offset_y, draw_direction,
				                                                self.corridor_draw_size)
				rect.mousePressEvent = partial(self.parent().parent().parent().on_corridor_press, room.room_from,
				                               room.room_to)
				rects.append(rect)
			self.drawn_rooms.append(f'{room.room_from}_{room.room_to}')
			
			if direction != '':
				offset_x, offset_y = __update_offsets(offset_x, offset_y, draw_direction,
				                                      room.length * self.corridor_draw_size)
				for other_room in [room.room_from, room.room_to]:
					if other_room not in self.drawn_rooms:
						other_rects, _, _ = self.get_rects(self.level.rooms[other_room], offset_x, offset_y, direction)
						rects.extend(other_rects)
			else:
				offset_x, offset_y = __update_offsets(offset_x, offset_y, draw_direction, corridor_offset)
				for (room_a, room_b) in [(room.room_from, room.room_to), (room.room_to, room.room_from)]:
					if room_a not in self.drawn_rooms:
						for new_direction in DIRECTIONS:
							if self.level.level_geometry[room_a][new_direction] == room_b:
								if new_direction == draw_direction:
									new_offset_x, new_offset_y = __update_offsets(offset_x, offset_y,
									                                              OPPOSITE_DIRECTIONS[new_direction],
									                                              room.length * self.corridor_draw_size / 2 + self.room_draw_size)
								else:
									new_offset_x, new_offset_y = __update_offsets(offset_x, offset_y,
									                                              OPPOSITE_DIRECTIONS[new_direction],
									                                              room.length * self.corridor_draw_size / 2)
								other_rects, _, _ = self.get_rects(self.level.rooms[room_a], new_offset_x, new_offset_y,
								                                   new_direction)
								rects.extend(other_rects)
		
		return rects, offset_x, offset_y
	
	def show_map_preview(self):
		self.drawn_rooms = []
		x, y = self.rect().x(), self.rect().y()
		self.corridor_draw_size = self.rect().height() * config.ui.minimap_corridor_scale
		self.room_draw_size = self.rect().height() * config.ui.minimap_room_scale
		
		self.scene.clear()
		self.scene.setBackgroundBrush(
			QBrush(QColor('#1e1d23' if self.parent().parent().parent().theme == 'DARK' else '#ececec')))
		
		if self.level.current_room != '':
			if self.level.current_room in self.level.rooms.keys():
				room = self.level.rooms[self.level.current_room]
			else:
				room = self.level.get_corridor(*self.level.current_room.split('-'))
			
			rects, _, _ = self.get_rects(room=room,
			                             offset_x=x // 2, offset_y=y // 2,
			                             direction='', selected=True)
			
			for rect in rects:
				self.scene.addItem(rect)


class ToolMode(Enum):
	USER = auto()
	LLM = auto()


class MainWindow(QMainWindow):
	def __init__(self, level: Level):
		super().__init__()
		self.level = level
		
		self.mode = ToolMode.LLM
		
		self.setObjectName("LLMaker")
		
		self.setWindowTitle("LLMaker Demo")
		self.resize(1280, 720)
		self.setWindowIcon(QIcon('assets/llmaker_logo.png'))
		
		self.main_ui_widget = QWidget(parent=self)
		
		self.theme = 'DARK'
		self.apply_theme()
		
		self.main_ui_layout = QHBoxLayout(self.main_ui_widget)
		
		self.previews = QGroupBox(parent=self.main_ui_widget)
		self.previews.setTitle('Previews')
		self.main_ui_layout.addWidget(self.previews, 3)
		
		self.previews_vertical_layout = QVBoxLayout(self.previews)
		
		self.room_label = QLabel(parent=self.previews)
		self.room_label.setText('<i>No current room</i>')
		self.previews_vertical_layout.addWidget(self.room_label)
		
		self.room_description = QLabel(parent=self.previews)
		self.room_description.setText('')
		self.previews_vertical_layout.addWidget(self.room_description)
		
		self.room_preview = RoomPreviewWidget(parent=self.previews, level=self.level)
		self.previews_vertical_layout.addWidget(self.room_preview, 8)
		
		self.map_label = QLabel(parent=self.previews)
		self.map_label.sizePolicy().setVerticalStretch(1)
		self.map_label.setText('Mission Map:')
		self.previews_vertical_layout.addWidget(self.map_label)
		
		self.map_preview = MapPreviewWidget(parent=self.previews, level=self.level)
		self.previews_vertical_layout.addWidget(self.map_preview, 2)
		
		self.actions_groupbox = QGroupBox(parent=self.main_ui_widget)
		self.actions_groupbox.setTitle('Chat History')
		self.main_ui_layout.addWidget(self.actions_groupbox, 1)
		
		self.actions_vertical_layout = QVBoxLayout(self.actions_groupbox)
		
		self.chat_area = ConversationWidget(parent=self.actions_groupbox)
		# self.chat_area.setPlaceholderText('Your conversation history will be displayed here...')
		# self.chat_area.setReadOnly(True)
		self.actions_vertical_layout.addWidget(self.chat_area, 8)
		
		self.chat_box = QLineEdit(parent=self.actions_groupbox)
		self.chat_box.setPlaceholderText('Type you message here, then press [Enter] to send it.')
		self.actions_vertical_layout.addWidget(self.chat_box, 1)
		
		self.pbar = QProgressBar(parent=self.actions_groupbox)
		self.pbar.setRange(0, 100)
		self.pbar.setHidden(True)
		self.actions_vertical_layout.addWidget(self.pbar)
		
		self.actions_buttons = []
		
		self.b_addRoom = QPushButton('Add Room')
		self.actions_buttons.append(self.b_addRoom)
		self.actions_vertical_layout.addWidget(self.b_addRoom)
		
		self.b_updateRoom = QPushButton('Update Room')
		self.actions_buttons.append(self.b_updateRoom)
		self.actions_vertical_layout.addWidget(self.b_updateRoom)
		
		self.b_removeRoom = QPushButton('Remove Room')
		self.actions_buttons.append(self.b_removeRoom)
		self.actions_vertical_layout.addWidget(self.b_removeRoom)
		
		self.b_addCorridor = QPushButton('Add Corridor')
		self.actions_buttons.append(self.b_addCorridor)
		self.actions_vertical_layout.addWidget(self.b_addCorridor)
		
		self.b_updateCorridor = QPushButton('Update Corridor')
		self.actions_buttons.append(self.b_updateCorridor)
		self.actions_vertical_layout.addWidget(self.b_updateCorridor)
		
		self.b_removeCorridor = QPushButton('Remove Corridor')
		self.actions_buttons.append(self.b_removeCorridor)
		self.actions_vertical_layout.addWidget(self.b_removeCorridor)
		
		self.b_addEnemy = QPushButton('Add Enemy')
		self.actions_buttons.append(self.b_addEnemy)
		self.actions_vertical_layout.addWidget(self.b_addEnemy)
		
		self.b_updateEnemy = QPushButton('Update Enemy')
		self.actions_buttons.append(self.b_updateEnemy)
		self.actions_vertical_layout.addWidget(self.b_updateEnemy)
		
		self.b_removeEnemy = QPushButton('Remove Enemy')
		self.actions_buttons.append(self.b_removeEnemy)
		self.actions_vertical_layout.addWidget(self.b_removeEnemy)
		
		self.b_addTrap = QPushButton('Add Trap')
		self.actions_buttons.append(self.b_addTrap)
		self.actions_vertical_layout.addWidget(self.b_addTrap)
		
		self.b_updateTrap = QPushButton('Update Trap')
		self.actions_buttons.append(self.b_updateTrap)
		self.actions_vertical_layout.addWidget(self.b_updateTrap)
		
		self.b_removeTrap = QPushButton('Remove Trap')
		self.actions_buttons.append(self.b_removeTrap)
		self.actions_vertical_layout.addWidget(self.b_removeTrap)
		
		self.b_addTreasure = QPushButton('Add Treasure')
		self.actions_buttons.append(self.b_addTreasure)
		self.actions_vertical_layout.addWidget(self.b_addTreasure)
		
		self.b_updateTreasure = QPushButton('Update Treasure')
		self.actions_buttons.append(self.b_updateTreasure)
		self.actions_vertical_layout.addWidget(self.b_updateTreasure)
		
		self.b_removeTreasure = QPushButton('Remove Treasure')
		self.actions_buttons.append(self.b_removeTreasure)
		self.actions_vertical_layout.addWidget(self.b_removeTreasure)
		
		# TODO: create qactions/qslots via dialogs for action_buttons
		
		for b in self.actions_buttons:
			b.hide()
		
		self.setCentralWidget(self.main_ui_widget)
		
		self.menuFile = self.menuBar().addMenu('&File')
		self.menuOptions = self.menuBar().addMenu('&Options')
		self.menuHelp = self.menuBar().addMenu('&Help')
		
		# Actions
		self.actionSave = QAction('Save', parent=self)
		self.actionSave.setToolTip('Save the current level design.')
		self.menuFile.addAction(self.actionSave)
		
		self.actionLoad = QAction('Load', parent=self)
		self.actionLoad.setToolTip('Load a saved level design.')
		self.menuFile.addAction(self.actionLoad)
		
		self.actionClear = QAction('Clear', parent=self)
		self.actionClear.setToolTip('Clear the current level and dialogue.')
		self.menuFile.addAction(self.actionClear)
		
		self.actionSwitchMode = QAction(f'Switch to {"LLM" if self.mode == ToolMode.USER else "USER"} mode',
		                                parent=self)
		self.actionSwitchMode.setToolTip(f'Switch LLMaker to {"LLM" if self.mode == ToolMode.USER else "USER"} mode.')
		self.menuOptions.addAction(self.actionSwitchMode)
		
		self.actionSwitchTheme = QAction(f'Switch to {"Light" if self.theme == "DARK" else "Dark"} theme',
		                                 parent=self)
		self.actionSwitchTheme.setToolTip(f'Switch LLMaker to {"Light" if self.theme == "DARK" else "Dark"} theme.')
		self.menuOptions.addAction(self.actionSwitchTheme)
		
		self.actionAbout = QAction('About', parent=self)
		self.actionAbout.setToolTip('About LLMaker')
		self.menuHelp.addAction(self.actionAbout)
		
		self.menuBar().addAction(self.menuFile.menuAction())
		self.menuBar().addAction(self.menuOptions.menuAction())
		self.menuBar().addAction(self.menuHelp.menuAction())
		
		self.chat_box.returnPressed.connect(self.process_user_input)
		self.actionSave.triggered.connect(self.save_level)
		self.actionLoad.triggered.connect(self.load_level)
		self.actionClear.triggered.connect(self.clear_level)
		self.actionSwitchMode.triggered.connect(self.switch_mode)
		self.actionSwitchTheme.triggered.connect(self.switch_theme)
		self.actionAbout.triggered.connect(self.show_about_dialog)
		
		self.chat_box.setFocus()
	
	@pyqtSlot(int)
	def update_progress(self, progress):
		logging.getLogger().debug(f'update_progress Task progress: {progress}')
		self.pbar.setValue(progress)
	
	@pyqtSlot(str)
	def handle_result(self, result):
		logging.getLogger().debug(f'handle_result Received LLM response')
		self.chat_area.add_message(result)
	
	@pyqtSlot()
	def task_finished(self):
		logging.getLogger().debug(f'task_finished Exchange finished')
		self.chat_box.setDisabled(False)
		self.chat_box.setFocus()
		self.pbar.reset()
		self.pbar.setHidden(True)
		self.update()
	
	@pyqtSlot()
	def process_user_input(self):
		user_input = self.chat_box.text()
		conversation_history = self.chat_area.get_conversation()
		
		logging.getLogger().debug(f'process_user_input Received input')
		
		self.chat_box.clear()
		self.chat_area.add_message(user_input)
		
		self.chat_box.setDisabled(True)
		
		logging.getLogger().debug(f'process_user_input Starting separate thread')
		
		self.worker = InputProcessor(self.level, user_input, conversation_history)
		self.thread = QThread()
		
		self.worker.moveToThread(self.thread)
		
		# Connect signals and slots
		self.thread.started.connect(self.worker.run)
		self.worker.result.connect(self.handle_result)
		self.worker.finished.connect(self.task_finished)
		self.worker.progress.connect(self.update_progress)
		self.worker.finished.connect(self.thread.quit)
		self.worker.finished.connect(self.worker.deleteLater)
		self.thread.finished.connect(self.thread.deleteLater)
		
		self.pbar.setHidden(False)
		self.pbar.reset()
		self.actions_groupbox.update()
		
		# Start the thread
		self.thread.start()
	
	def paintEvent(self, a0):
		if self.level.current_room:
			if self.level.current_room in self.level.rooms.keys():
				self.room_label.setText(f'Room: <b><i>{self.level.current_room}</i></b>')
				self.room_description.setText(f'<i>{self.level.rooms[self.level.current_room].description}</i>')
			else:
				corridor = self.level.get_corridor(*self.level.current_room.split('-'))
				self.room_label.setText(
					f'Corridor between <b><i>{corridor.room_from}</i></b> and <b><i>{corridor.room_to}</i></b>')
				self.room_description.setText('')
		else:
			self.room_label.setText('<i>No current room</i>')
			self.room_description.setText('')
	
	def on_room_press(self, room_name, event):
		self.level.current_room = room_name
		self.update()
	
	def on_corridor_press(self, room_from_name, room_to_name, event):
		corridor = self.level.get_corridor(room_from_name, room_to_name)
		self.level.current_room = corridor.name
		self.update()
	
	@pyqtSlot()
	def save_level(self):
		try:
			tmp_filename, _ = QFileDialog.getSaveFileName(self,
			                                              caption="Save Level",
			                                              directory=config.levels_dir,
			                                              filter="All Files(*);;Binary Files(*.bin)")
			if tmp_filename:
				assert len(self.level.rooms) > 0, 'Can\'t save an empty level!'
				self.level.save_to_file(filename=tmp_filename,
				                        conversation=self.chat_area.get_conversation())
				
				dlg = QMessageBox(self)
				dlg.setWindowTitle("LLMaker Message")
				dlg.setText(f"The level has been successfully saved to <i>{os.path.split(tmp_filename)[1]}</i>!")
				_ = dlg.exec()
		except Exception as e:
			dlg = QErrorMessage(self)
			dlg.setWindowTitle("LLMaker Error")
			dlg.showMessage(str(e))
			_ = dlg.exec()
	
	@pyqtSlot()
	def load_level(self):
		tmp_filename, _ = QFileDialog.getOpenFileName(self,
		                                              caption="Load Level",
		                                              directory=config.levels_dir,
		                                              filter="All Files(*);;Binary Files(*.bin)")
		
		if tmp_filename:
			try:
				level, conversation = Level.load_from_file(tmp_filename)
				self.set_level(level)
				
				for i, line in enumerate(conversation.split('\n')):
					line = line.replace('You: ', '').replace('AI: ', '')
					self.chat_area.add_message(line)
				
				dlg = QMessageBox(self)
				dlg.setWindowTitle("LLMaker Message")
				dlg.setText(f"The level has been successfully loaded!")
				_ = dlg.exec()
				
				self.update()
			except Exception as e:
				dlg = QErrorMessage(self)
				dlg.setWindowTitle("LLMaker Error")
				dlg.showMessage(str(e))
				_ = dlg.exec()
	
	@pyqtSlot()
	def clear_level(self):
		self.set_level(Level())
		self.chat_box.clear()
		self.chat_area.reset()
		self.update()
	
	@pyqtSlot()
	def switch_mode(self):
		self.actionSwitchMode.setText(f'Switch to {"USER" if self.mode == ToolMode.USER else "LLM"} mode')
		self.actionSwitchMode.setToolTip(f'Switch LLMaker to {"USER" if self.mode == ToolMode.USER else "LLM"} mode.')
		self.mode = ToolMode.USER if self.mode == ToolMode.LLM else ToolMode.LLM
		if self.mode == ToolMode.USER:
			self.chat_area.hide()
			self.chat_box.hide()
			self.actions_groupbox.setTitle('Available Commands')
			for b in self.actions_buttons:
				b.show()
		else:
			self.chat_area.show()
			self.chat_box.show()
			self.actions_groupbox.setTitle('Chat History')
			for b in self.actions_buttons:
				b.hide()
		logging.info(f'Switched mode to {"USER" if self.mode == ToolMode.USER else "LLM"}')
		self.update()
	
	@pyqtSlot()
	def switch_theme(self):
		self.actionSwitchTheme.setText(f'Switch to {"Light" if self.theme == "LIGHT" else "Dark"} theme')
		self.actionSwitchMode.setToolTip(f'Switch LLMaker to {"Light" if self.theme == "LIGHT" else "Dark"} theme.')
		self.theme = 'DARK' if self.theme == 'LIGHT' else 'LIGHT'
		self.apply_theme()
		self.update()
		logging.info(f'Switched theme to {"Light" if self.theme == "LIGHT" else "Dark"}')
	
	@pyqtSlot()
	def show_about_dialog(self):
		QMessageBox.about(self, 'About LLMaker',
		                  f'LLMaker v0.1\nIEEE Conference on Games 2024 Demo\n\nDeveloped by: Roberto Gallota (Institute of Digital Games, University of Malta)')
	
	def set_level(self, level: Level):
		self.level = level
		self.room_preview.level = level
		self.map_preview.level = level
	
	def apply_theme(self):
		try:
			with open(f'assets/themes/stylesheet_{self.theme.lower()}.css', 'r') as f:
				self.setStyleSheet(f.read())
		except FileNotFoundError:
			raise ValueError(f'Unknown theme: {self.theme}')
