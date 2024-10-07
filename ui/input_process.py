import logging
import time
from typing import List, Union, Any, Dict

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QDialog
from gptfunctionutil import LibCommand

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from llm_backend import chat_llm
from sd_backend import generate_room, generate_entity, generate_corridor
from utils import compute_level_diffs, process_diff


class UIInputProcessor(QObject):
	finished = pyqtSignal()
	progress = pyqtSignal(int)
	result = pyqtSignal(str)
	
	def __init__(self,
	             level: Level,
	             user_input: str,
	             conversation_history: str):
		super(UIInputProcessor, self).__init__()
		self.level = level
		self.user_input = user_input
		self.conversation_history = conversation_history
	
	def run(self) -> str:
		self.progress_n = 0
		ai_response = chat_llm(user_message=self.user_input,
		                       conversation_history=self.conversation_history,
		                       level=self.level)
		self.result.emit(ai_response)
		
		to_process, additional_data = compute_level_diffs(level=self.level)
		
		progress_delta = int((1 / (1 + len(to_process))) * 100)
		
		self.progress_n += progress_delta
		self.progress.emit(self.progress_n)
		
		for i, obj in enumerate(to_process):
			process_diff(obj, additional_data[i])
			self.progress_n += progress_delta
			self.progress.emit(self.progress_n)
		
		time.sleep(0.5)
		self.finished.emit()


class DebugInputProcessor(QObject):
	finished = pyqtSignal(str)
	progress = pyqtSignal(int)
	
	def __init__(self,
	             kwargs: Dict[str, Any],
	             func: LibCommand,
	             dialog: QDialog):
		super(DebugInputProcessor, self).__init__()
		self.kwargs = kwargs
		self.func = func
		self.dialog = dialog
		
	def run(self) -> str:
		self.progress_n = 0
		
		try:
			submission_output = self.func.command(**self.kwargs)
		
			to_process, additional_data = compute_level_diffs(level=self.kwargs['level'])
			
			progress_delta = int((1 / (1 + len(to_process))) * 100)
			
			self.progress_n += progress_delta
			self.progress.emit(self.progress_n)
			
			for i, obj in enumerate(to_process):
				process_diff(obj, additional_data[i])
				self.progress_n += progress_delta
				self.progress.emit(self.progress_n)
			
			time.sleep(0.5)
			self.finished.emit(submission_output)
		except Exception as e:
			print(e)
			print(e.__traceback__)
			QMessageBox.critical(self.dialog, "Error", str(e))