import copy
import logging
import os

from PyQt6.QtCore import QThread, pyqtSlot, QThreadPool
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import QErrorMessage, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, \
	QMessageBox, QProgressBar, QPushButton, QSplashScreen, \
	QVBoxLayout, QWidget, QMenu
from dungeon_despair.domain.utils import make_corridor_name, get_enum_by_value

from chat_message import Conversation
from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.scenario import check_level_playability, ScenarioType
from dungeon_despair.functions import DungeonCrawlerFunctions
from freyr_llm import get_freyr_model, LLMsCache
from tool_llm import get_tool_model
from ui.chat import ConversationWidget
from ui.dyn_dialog import AddEntityDialog, DebugFunctionsDialog, UpdateEntityDialog, RemoveEntityDialog, check_available_action, function_to_dialog
from ui.encounter_preview import EncounterPreviewWidget
from ui.input_process import UIInputProcessor
from ui.map_preview import MapPreviewWidget
from utils import LLMMode, ToolMode, ThemeMode
from versioning import VersionHandler
from configs import resource_path


def get_splash_screen():
	pixmap = QPixmap(resource_path('assets/llmaker_splash.png'))
	splash = QSplashScreen(pixmap)
	return splash


class MainWindow(QMainWindow):
	def __init__(self, level: Level):
		super().__init__()
		self.level = level

		self.mode = get_enum_by_value(ToolMode, config.start_mode)
		self.llm_mode = get_enum_by_value(LLMMode, config.llm_mode)
		self.theme = get_enum_by_value(ThemeMode, config.theme)

		self.apply_theme()
		
		self.setObjectName("LLMaker")
		
		self.setWindowTitle("LLMaker")
		self.resize(1280, 720)
		self.setWindowIcon(QIcon(resource_path('assets/llmaker_logo.png')))
		
		self.main_ui_widget = QWidget(parent=self)
		
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
		
		self.room_preview = EncounterPreviewWidget(parent=self.previews, level=self.level)
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
		
		self.user_mode_area = QGroupBox(parent=self.actions_groupbox)
		self.user_mode_layout = QVBoxLayout(self.user_mode_area)

		self.user_mode_layout.addWidget(QLabel('Room edits:'))
		self.room_edits_widget = QWidget(parent=self.user_mode_area)
		self.room_edits_layout = QHBoxLayout(self.room_edits_widget)
		for btitle, funcname in zip(['Add', 'Update', 'Remove'],
							  		['create_room', 'update_room', 'remove_room']):
			button = QPushButton(btitle)
			button.clicked.connect(self.create_button_handler(DungeonCrawlerFunctions().FunctionDict[funcname], button))
			self.room_edits_layout.addWidget(button)
		self.user_mode_layout.addWidget(self.room_edits_widget)

		self.user_mode_layout.addWidget(QLabel('Corridor edits:'))
		self.corridor_edits_widget = QWidget(parent=self.user_mode_area)
		self.corridor_edits_layout = QHBoxLayout(self.corridor_edits_widget)
		for btitle, funcname in zip(['Add', 'Update', 'Remove'],
							  		['add_corridor', 'update_corridor', 'remove_corridor']):
			button = QPushButton(btitle)
			button.clicked.connect(self.create_button_handler(DungeonCrawlerFunctions().FunctionDict[funcname], button))
			self.corridor_edits_layout.addWidget(button)
		self.user_mode_layout.addWidget(self.corridor_edits_widget)
		
		self.user_mode_layout.addWidget(QLabel('Entity edits:'))
		self.entity_edits_widget = QWidget(parent=self.user_mode_area)
		self.entity_edits_layout = QHBoxLayout(self.entity_edits_widget)
		for btitle, bclass in zip(['Add', 'Edit', 'Remove'],
								  [AddEntityDialog, UpdateEntityDialog, RemoveEntityDialog]):
			button = QPushButton(btitle)
			button.clicked.connect(self.create_button_handler(func=None, button=button, dialogclass=bclass))
			self.entity_edits_layout.addWidget(button)
		self.user_mode_layout.addWidget(self.entity_edits_widget)

		self.user_mode_layout.addWidget(QLabel('Attacks edits:'))
		self.attack_edits_widget = QWidget(parent=self.user_mode_area)
		self.attack_edits_layout = QHBoxLayout(self.attack_edits_widget)
		for btitle, funcname in zip(['Add', 'Update', 'Remove'],
							  		['add_attack', 'update_attack', 'remove_attack']):
			button = QPushButton(btitle)
			button.clicked.connect(self.create_button_handler(DungeonCrawlerFunctions().FunctionDict[funcname], button))
			self.attack_edits_layout.addWidget(button)
		self.user_mode_layout.addWidget(self.attack_edits_widget)

		self.actions_vertical_layout.addWidget(self.user_mode_area, 8)

		self.chat_area = ConversationWidget(parent=self.actions_groupbox)
		self.actions_vertical_layout.addWidget(self.chat_area, 8)
		
		self.chat_box = QLineEdit(parent=self.actions_groupbox)
		self.chat_box.setPlaceholderText('Type you message here, then press [Enter] to send it.')
		self.actions_vertical_layout.addWidget(self.chat_box, 1)
		
		self.pbar = QProgressBar(parent=self.actions_groupbox)
		self.pbar.setRange(0, 100)
		self.pbar.setHidden(True)
		self.actions_vertical_layout.addWidget(self.pbar)
				
		self.setCentralWidget(self.main_ui_widget)
		
		self.menuFile = self.menuBar().addMenu('&File')
		self.menuOptions = self.menuBar().addMenu('&Options')
		self.menuEdit = self.menuBar().addMenu('&Edit')
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
		
		self.menuFile.addSeparator()
		
		self.actionExport = QAction('Export', parent=self)
		self.actionExport.setToolTip('Finalize and export the current level as scenario.')
		self.menuFile.addAction(self.actionExport)
		
		self.actionSwitchMode = QAction(f'Switch to {"LLM" if self.mode == ToolMode.USER else "USER"} mode',
		                                parent=self)
		self.actionSwitchMode.setToolTip(f'Switch LLMaker to {"LLM" if self.mode == ToolMode.USER else "USER"} mode.')
		if config.can_switch_mode:
			self.menuOptions.addAction(self.actionSwitchMode)
		
		self.actionSwitchTheme = QAction(f'Switch to {"Light" if self.theme == ThemeMode.DARK else "Dark"} theme',
		                                 parent=self)
		self.actionSwitchTheme.setToolTip(
			f'Switch LLMaker to {"Light" if self.theme == ThemeMode.DARK else "Dark"} theme.')
		self.menuOptions.addAction(self.actionSwitchTheme)

		self.menuOptions.addSeparator()

		self.llm_menu = self.menuOptions.addMenu('LLMs')

		self.llm_mode_action = QAction(f'Use {LLMMode.TOOL.value if self.llm_mode == LLMMode.FREYR else LLMMode.FREYR.value} mode', parent=self.llm_menu)
		self.llm_mode_action.triggered.connect(self.toggle_llm_mode)
		self.llm_menu.addAction(self.llm_mode_action)

		self.llm_menu.addSeparator()

		self.freyr_menu = self.llm_menu.addMenu('Freyr')
		self.freyr_intent = self.freyr_menu.addMenu('Intent')
		self.freyr_params = self.freyr_menu.addMenu('Parameters')
		self.freyr_chat = self.freyr_menu.addMenu('Chat')
		self.freyr_summary = self.freyr_menu.addMenu('Summary')

		self.llm_menu.addSeparator()
		self.tool_menu = self.llm_menu.addMenu('Tools')
		self.tool_model = self.tool_menu.addMenu('Model')

		for submenu, role in zip([self.freyr_intent, self.freyr_params, self.freyr_chat, self.freyr_summary],
						   		 ['intent', 'params', 'chat', 'summary']):
			for available_llm in LLMsCache.get_ollama_models():
				llm_choice = QAction(available_llm, parent=submenu, checkable=True)
				if self.llm_mode == LLMMode.FREYR and get_freyr_model().cache.get_model_by_role(role) == available_llm:
					llm_choice.setChecked(True)
				llm_choice.triggered.connect(self.create_freyr_models_handler(role, submenu, llm_choice))
				submenu.addAction(llm_choice)
		# TODO: Would make more sense to have this ONLY when in TOOL mode
		for available_llm in LLMsCache.get_ollama_models():
			llm_choice = QAction(available_llm, parent=self.tool_model, checkable=True)
			if self.llm_mode == LLMMode.TOOL and available_llm == get_tool_model().model_name:
				llm_choice.setChecked(True)
			llm_choice.triggered.connect(self.create_tool_models_handler(self.tool_model, llm_choice))
			self.tool_model.addAction(llm_choice)

		self.actionUndo = QAction('Undo', parent=self)
		self.actionUndo.setToolTip('Undo latest change')
		self.menuEdit.addAction(self.actionUndo)
		self.actionRedo = QAction('Redo', parent=self)
		self.actionRedo.setToolTip('Redo latest change')
		self.menuEdit.addAction(self.actionRedo)

		self.actionAbout = QAction('About', parent=self)
		self.actionAbout.setToolTip('About LLMaker')
		self.menuHelp.addAction(self.actionAbout)
		
		self.menuBar().addAction(self.menuFile.menuAction())
		self.menuBar().addAction(self.menuOptions.menuAction())
		self.menuBar().addAction(self.menuEdit.menuAction())
		self.menuBar().addAction(self.menuHelp.menuAction())
		
		self.chat_box.returnPressed.connect(self.process_user_input)
		self.actionSave.triggered.connect(self.save_level)
		self.actionLoad.triggered.connect(self.load_level)
		self.actionClear.triggered.connect(self.clear_level)
		self.actionExport.triggered.connect(self.export_level)
		self.actionSwitchMode.triggered.connect(self.switch_mode)
		self.actionSwitchTheme.triggered.connect(self.switch_theme)
		self.actionUndo.triggered.connect(self.undo_edit)
		self.actionRedo.triggered.connect(self.redo_edit)
		self.actionAbout.triggered.connect(self.show_about_dialog)
		
		self.threadpool = QThreadPool()

		self.switch_mode(keep=True)
		self.validate_actions_buttons()
		self.versioning = VersionHandler(level=self.level,
										 chat=self.chat_area.conversation.messages)
		
		self.chat_box.setFocus()
	
	def create_button_handler(self, func, button, dialogclass=None):
		button.setProperty('dialogclass', dialogclass if dialogclass is not None else function_to_dialog[func.internal_name])
		def handler():
			if not dialogclass:
				dialog = function_to_dialog[func.internal_name](self.level, func, button)
			else:
				dialog = dialogclass(self.level, None, button)
			dialog.exec()
			self.versioning.commit(self.level, self.chat_area.conversation.messages)
			self.validate_actions_buttons()
		
		return handler

	def validate_actions_buttons(self) -> None:
		for container_widget in [self.room_edits_widget, self.corridor_edits_widget, self.entity_edits_widget, self.attack_edits_widget]:
			for btn in container_widget.children()[1:]:
				btn.setDisabled(not check_available_action(self.level, btn.property('dialogclass')))

	def create_freyr_models_handler(self, role: str, menu: QMenu, action: QAction):
		def handler():
			freyr_instance = get_freyr_model()
			if freyr_instance.cache.get_model_by_role(role) != action.text():
				for other_action in menu.actions():
					other_action.setChecked(False)
				action.setChecked(True)
				freyr_instance.cache.drop_model_by_role(role)
				freyr_instance.cache.try_add_model(role=role, model_name=action.text())
		
		return handler				
	
	def create_tool_models_handler(self, menu: QMenu, action: QAction):
		def handler():
			tool_instance = get_tool_model()
			if tool_instance.model_name != action.text():
				for other_action in menu.actions():
					other_action.setChecked(False)
				action.setChecked(True)
				tool_instance.model_name = action.text()

		return handler

	def update_progress(self, progress):
		logging.getLogger('llmaker').debug(f'MainWindow.update_progress Task progress: {progress}')
		self.pbar.setValue(progress)
	
	def handle_result(self, result):
		logging.getLogger('llmaker').debug(f'MainWindow.handle_result Received LLM response')
		self.chat_area.add_message(result, role='them')
	
	def task_error(self, err_data):
		QMessageBox.critical(self, f'LLMaker Error: {err_data[0].__name__}', str(err_data[1]))

	def task_finished(self):
		logging.getLogger('llmaker').debug(f'MainWindow.task_finished Exchange finished')
		self.chat_box.setDisabled(False)
		self.chat_box.setFocus()
		self.pbar.reset()
		self.pbar.setHidden(True)
		# Note: This commits every time a message is sent, regardless of the operation carried out
		self.versioning.commit(self.level,
						 	   self.chat_area.conversation.messages)
		self.level.save_to_file(filename=os.path.join(config.levels_dir, config.tmp_level),
						  		conversation=self.chat_area.conversation.to_json())
		self.chat_area.update()
		if self.mode == ToolMode.USER:
			self.validate_actions_buttons()
		self.update()
	
	@pyqtSlot()
	def process_user_input(self):
		user_input = self.chat_box.text()
		conversation_history = self.chat_area.get_conversation()
		
		logging.getLogger('llmaker').debug(f'MainWindow.process_user_input Received input')
		
		self.chat_box.clear()
		self.chat_area.add_message(user_input, role='me')
		self.chat_area.update()
		
		self.chat_box.setDisabled(True)
		
		logging.getLogger('llmaker').debug(f'MainWindow.process_user_input Starting separate thread')
		
		worker = UIInputProcessor(self.level, user_input, conversation_history, self.llm_mode)
		
		worker.signals.result.connect(self.handle_result)
		worker.signals.error.connect(self.task_error)
		worker.signals.progress.connect(self.update_progress)
		worker.signals.finished.connect(self.task_finished)
				
		self.pbar.setHidden(False)
		self.pbar.reset()
		self.actions_groupbox.update()
		self.chat_area.update()
		
		# Start the thread
		self.threadpool.start(worker)
	
	def paintEvent(self, a0):
		if self.level.current_room:
			if self.level.current_room in self.level.rooms.keys():
				self.room_label.setText(f'Room: <b><i>{self.level.current_room}</i></b>')
				self.room_description.setText(f'<i>{self.level.rooms[self.level.current_room].description}</i>')
			else:
				corridor = self.level.corridors[self.level.current_room]
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
		corridor = self.level.corridors[make_corridor_name(room_from_name=room_from_name, room_to_name=room_to_name)]
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
				                        conversation=self.chat_area.conversation.to_json())
				
				dlg = QMessageBox(self)
				dlg.setWindowTitle("LLMaker Message")
				dlg.setText(f"The level has been successfully saved to <i>{os.path.basename(tmp_filename)}</i>!")
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
				level, conversation_json = Level.load_from_file(tmp_filename)
							
				conversation = Conversation.from_json(conversation_json)
				for msg in conversation.messages:
					self.chat_area.add_message(msg.content, role=msg.role)
				
				self.set_level(level)
				self.versioning = VersionHandler(self.level, conversation.messages)
				QMessageBox.information(self, "LLMaker Message", f"The level {tmp_filename} has been successfully loaded!")
				self.update()
				self.chat_area.update()
				if self.mode == ToolMode.USER:
					self.validate_actions_buttons()
			except Exception as e:
				dlg = QErrorMessage(self)
				dlg.setWindowTitle("LLMaker Error")
				dlg.showMessage(str(e))
				_ = dlg.exec()
	
	@pyqtSlot()
	def clear_level(self):
		self.set_level(Level())
		self.versioning = VersionHandler(level=self.level, chat=[])
		self.chat_box.clear()
		self.chat_area.reset()
		if self.mode == ToolMode.USER:
			self.validate_actions_buttons()
		self.update()
	
	@pyqtSlot()
	def export_level(self):
		try:
			tmp_filename, _ = QFileDialog.getSaveFileName(self,
			                                              caption="Export Level as Scenario",
			                                              directory=config.scenarios_dir,
			                                              filter="All Files(*);;Binary Files(*.bin)")
			if tmp_filename:
				if check_level_playability(self.level, ScenarioType.EXPLORE):
					self.level.export_level_as_scenario(filename=tmp_filename)
					QMessageBox.information(self, "LLMaker Message", f"The level has been successfully exported as scenario to <i>{os.path.basename(tmp_filename)}</i>!")
		except Exception as e:
			dlg = QErrorMessage(self)
			dlg.setWindowTitle("LLMaker Error")
			dlg.showMessage(str(e))
			_ = dlg.exec()
	
	@pyqtSlot()
	def undo_edit(self):
		if self.versioning.can_undo:
			prev_level, prev_chat = self.versioning.undo()
			self.set_level(prev_level)
			self.chat_area.reset()
			for msg in prev_chat:
				self.chat_area.add_message(msg.content, msg.role)
			self.update()
			self.chat_area.update()
			if self.mode == ToolMode.USER:
				self.validate_actions_buttons()
		else:
			QMessageBox.warning(self, "LLMaker Warning", "No undos available!")
	
	@pyqtSlot()
	def redo_edit(self):
		if self.versioning.can_redo:
			next_level, next_chat =	self.versioning.redo()
			self.set_level(next_level)
			self.chat_area.reset()
			for msg in next_chat:
				self.chat_area.add_message(msg.content, msg.role)
			self.update()
			self.chat_area.update()
			if self.mode == ToolMode.USER:
				self.validate_actions_buttons()
		else:
			QMessageBox.warning(self, "LLMaker Warning", "No redos available!")

	@pyqtSlot()
	def switch_mode(self,
				 	keep: bool = False):
		if not keep:
			self.mode = ToolMode.USER if self.mode == ToolMode.LLM else ToolMode.LLM
		self.actionSwitchMode.setText(f'Switch to {"USER" if self.mode == ToolMode.LLM else "LLM"} mode')
		self.actionSwitchMode.setToolTip(f'Switch LLMaker to {"USER" if self.mode == ToolMode.LLM else "LLM"} mode.')
		if self.mode == ToolMode.USER:
			self.chat_area.hide()
			self.chat_box.hide()
			self.actions_groupbox.setTitle('Available Commands')
			self.user_mode_area.show()
			self.validate_actions_buttons()
		else:
			self.chat_area.show()
			self.chat_box.show()
			self.actions_groupbox.setTitle('Chat History')
			self.user_mode_area.hide()
		logging.info(f'MainWindow.switch_mode Switched mode to {"USER" if self.mode == ToolMode.USER else "LLM"}')
		self.update()
	
	@pyqtSlot()
	def switch_theme(self):
		self.actionSwitchTheme.setText(f'Switch to {"Light" if self.theme == ThemeMode.LIGHT else "Dark"} theme')
		self.actionSwitchMode.setToolTip(
			f'Switch LLMaker to {"Light" if self.theme == ThemeMode.LIGHT else "Dark"} theme.')
		self.theme = ThemeMode.DARK if self.theme == ThemeMode.LIGHT else ThemeMode.LIGHT
		self.apply_theme()
		self.update()
		logging.info(f'MainWindow.switch_theme Switched theme to {"Light" if self.theme == ThemeMode.LIGHT else "Dark"}')
	
	@pyqtSlot()
	def show_about_dialog(self):
		QMessageBox.about(self, 'About LLMaker',
		                  f'LLMaker Client v0.1\n\nDeveloped by: Roberto Gallota (Institute of Digital Games, University of Malta)')
	
	def set_level(self, level: Level):
		self.level = level
		self.room_preview.level = level
		self.map_preview.level = level
	
	def apply_theme(self):
		try:
			with open(resource_path(f'assets/themes/stylesheet_{self.theme.value}.css'), 'r') as f:
				self.setStyleSheet(f.read())
		except FileNotFoundError:
			raise ValueError(f'Unknown theme: {self.theme.name}')
		
	@pyqtSlot()
	def toggle_llm_mode(self):
		self.llm_mode = LLMMode.FREYR if self.llm_mode == LLMMode.TOOL else LLMMode.TOOL
		self.llm_mode_action.setText(f'Use {LLMMode.TOOL.value if self.llm_mode == LLMMode.FREYR else LLMMode.FREYR.value} mode')