import logging
import os

from PyQt6.QtCore import QThread, pyqtSlot
from PyQt6.QtGui import QAction, QIcon, QPixmap
from PyQt6.QtWidgets import QErrorMessage, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, \
	QMessageBox, QProgressBar, QPushButton, QSplashScreen, \
	QVBoxLayout, QWidget

from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.scenario import check_level_playability, ScenarioType
from dungeon_despair.functions import DungeonCrawlerFunctions
from ui.chat import ConversationWidget
from ui.dyn_dialog import DebugFunctionsDialog
from ui.encounter_preview import EncounterPreviewWidget
from ui.input_process import UIInputProcessor
from ui.map_preview import MapPreviewWidget
from utils import ToolMode, ThemeMode


def get_splash_screen():
	pixmap = QPixmap('assets/llmaker_splash.png')
	splash = QSplashScreen(pixmap)
	return splash


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
		self.actionExport.triggered.connect(self.export_level)
		self.actionSwitchMode.triggered.connect(self.switch_mode)
		self.actionSwitchTheme.triggered.connect(self.switch_theme)
		self.actionAbout.triggered.connect(self.show_about_dialog)
		
		self.switch_mode()
		
		self.chat_box.setFocus()
	
	def create_button_handler(self, func, button):
		def handler():
			dialog = DebugFunctionsDialog(self.level, func, button)
			dialog.exec()
		
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
		
		self.worker = UIInputProcessor(self.level, user_input, conversation_history)
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
