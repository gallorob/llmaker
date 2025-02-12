import copy
import logging
import os

from PyQt6.QtCore import QThread, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import QErrorMessage, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, \
	QMessageBox, QProgressBar, QPushButton, QSplashScreen, \
	QVBoxLayout, QWidget, QMenu
from dungeon_despair.domain.utils import make_corridor_name

from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.scenario import check_level_playability, ScenarioType
from dungeon_despair.functions import DungeonCrawlerFunctions
from freyr_llm import LLMsCache, freyr_model
from tool_llm import tool_model
from ui.chat import ConversationWidget
from ui.dyn_dialog import DebugFunctionsDialog
from ui.encounter_preview import EncounterPreviewWidget
from ui.input_process import UIInputProcessor
from ui.map_preview import MapPreviewWidget
from utils import LLMMode, ToolMode, ThemeMode


def get_splash_screen():
	pixmap = QPixmap('assets/llmaker_splash.png')
	splash = QSplashScreen(pixmap)
	return splash


class MainWindow(QMainWindow):
	def __init__(self, level: Level):
		super().__init__()
		self.level = level
		self.levels_hist = [copy.deepcopy(self.level)]
		self.level_idx = 0
		
		self.mode = ToolMode.LLM
		self.llm_mode = LLMMode.FREYR
		
		self.setObjectName("LLMaker")
		
		self.setWindowTitle("LLMaker Demo")
		self.resize(1280, 720)
		self.setWindowIcon(QIcon('assets/llmaker_logo.png'))
		
		self.main_ui_widget = QWidget(parent=self)
		
		self.theme = ThemeMode.DARK
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
		for k, func in DungeonCrawlerFunctions().FunctionDict.items():
			button = QPushButton(k)
			button.clicked.connect(self.create_button_handler(func, button))
			button.hide()
			self.actions_buttons.append(button)
			self.actions_vertical_layout.addWidget(button)
		
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
				if self.llm_mode == LLMMode.FREYR and freyr_model.cache.get_model_by_role(role) == available_llm:
					llm_choice.setChecked(True)
				llm_choice.triggered.connect(self.create_freyr_models_handler(role, submenu, llm_choice))
				submenu.addAction(llm_choice)
		
		for available_llm in LLMsCache.get_ollama_models():
			llm_choice = QAction(available_llm, parent=self.tool_model, checkable=True)
			if self.llm_mode == LLMMode.TOOL and available_llm == tool_model.model_name:
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
		
		# self.switch_mode()
		
		self.chat_box.setFocus()
	
	def create_button_handler(self, func, button):
		def handler():
			if self.level_idx < len(self.levels_hist) - 1:
				self.levels_hist = self.levels_hist[:self.level_idx + 1]

			dialog = DebugFunctionsDialog(self.level, func, button)
			dialog.exec()

			self.level_idx += 1
			self.levels_hist.append(copy.deepcopy(self.level))
		
		return handler

	def create_freyr_models_handler(self, role: str, menu: QMenu, action: QAction):
		def handler():
			# Action is checked before the handler is called, so we have to check LLMs cache to see if the user is switching models
			if freyr_model.cache.get_model_by_role(role) != action.text():
				for other_action in menu.actions():
					other_action.setChecked(False)
				action.setChecked(True)
				freyr_model.cache.drop_model_by_role(role)
				freyr_model.cache.try_add_model(role=role, model_name=action.text())
		
		return handler				
	
	def create_tool_models_handler(self, menu: QMenu, action: QAction):
		def handler():
			if tool_model.model_name != action.text():
				for other_action in menu.actions():
					other_action.setChecked(False)
				action.setChecked(True)
				tool_model.model_name = action.text()

		return handler

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
		self.level_idx += 1
		self.levels_hist.append(copy.deepcopy(self.level))
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
		
		if self.level_idx < len(self.levels_hist) - 1:
			self.levels_hist = self.levels_hist[:self.level_idx + 1]

		self.worker = UIInputProcessor(self.level, user_input, conversation_history, self.llm_mode)
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
				                        conversation=self.chat_area.get_conversation())
				
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
				level, conversation = Level.load_from_file(tmp_filename)
				
				self.levels_hist = [copy.deepcopy(level)]
				self.level_idx = 0
				
				self.set_level(level)
				
				for i, line in enumerate(conversation.split('\n\n\n')):
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
	def export_level(self):
		try:
			tmp_filename, _ = QFileDialog.getSaveFileName(self,
			                                              caption="Save Level",
			                                              directory=config.scenarios_dir,
			                                              filter="All Files(*);;Binary Files(*.bin)")
			if tmp_filename:
				
				if check_level_playability(self.level, ScenarioType.EXPLORE):
					self.level.export_level_as_scenario(filename=tmp_filename)
				
					dlg = QMessageBox(self)
					dlg.setWindowTitle("LLMaker Message")
					dlg.setText(f"The level has been successfully exported as scenario to <i>{os.path.basename(tmp_filename)}</i>!")
					_ = dlg.exec()
		except Exception as e:
			dlg = QErrorMessage(self)
			dlg.setWindowTitle("LLMaker Error")
			dlg.showMessage(str(e))
			_ = dlg.exec()
	
	@pyqtSlot()
	def undo_edit(self):
		if self.level_idx > 0:
			self.level_idx -= 1
			self.set_level(copy.deepcopy(self.levels_hist[self.level_idx]))
			# TODO: Should also handle chat messages
			self.update()
		else:
			QMessageBox.warning(None, "LLMaker Warning", "No undos available!")
	
	@pyqtSlot()
	def redo_edit(self):
		if self.level_idx < len(self.levels_hist) - 1:
			self.level_idx += 1
			self.set_level(copy.deepcopy(self.levels_hist[self.level_idx]))
			# TODO: Should also handle chat messages
			self.update()
		else:
			QMessageBox.warning(None, "LLMaker Warning", "No redos available!")

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
		self.actionSwitchTheme.setText(f'Switch to {"Light" if self.theme == ThemeMode.LIGHT else "Dark"} theme')
		self.actionSwitchMode.setToolTip(
			f'Switch LLMaker to {"Light" if self.theme == ThemeMode.LIGHT else "Dark"} theme.')
		self.theme = ThemeMode.DARK if self.theme == ThemeMode.LIGHT else ThemeMode.LIGHT
		self.apply_theme()
		self.update()
		logging.info(f'Switched theme to {"Light" if self.theme == ThemeMode.LIGHT else "Dark"}')
	
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
			with open(f'assets/themes/stylesheet_{self.theme.value}.css', 'r') as f:
				self.setStyleSheet(f.read())
		except FileNotFoundError:
			raise ValueError(f'Unknown theme: {self.theme.name}')
		
	@pyqtSlot()
	def toggle_llm_mode(self):
		self.llm_mode = LLMMode.FREYR if self.llm_mode == LLMMode.TOOL else LLMMode.TOOL
		self.llm_mode_action.setText(f'Use {LLMMode.TOOL.value if self.llm_mode == LLMMode.FREYR else LLMMode.FREYR.value} mode')