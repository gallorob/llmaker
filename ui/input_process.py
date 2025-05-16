import logging
from random import random
import time
from typing import List, Any, Dict

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QDialog
from gptfunctionutil import LibCommand

from dungeon_despair.domain.level import Level
from chat_message import ChatMessage
from utils import LLMMode, compute_level_diffs, process_diff
from freyr_llm import get_freyr_model
from tool_llm import get_tool_model


class UIInputProcessor(QObject):
	finished = pyqtSignal()
	progress = pyqtSignal(int)
	result = pyqtSignal(str)
	
	def __init__(self,
	             level: Level,
	             user_input: str,
	             conversation_history: List[ChatMessage],
				 llm_mode: LLMMode):
		super(UIInputProcessor, self).__init__()
		self.level = level
		self.user_input = user_input
		self.conversation_history = conversation_history
		self.mode = llm_mode
	
	def run(self) -> str:
		self.progress_n = 0
		
		if self.mode == LLMMode.FREYR: m = get_freyr_model
		elif self.mode == LLMMode.TOOL: m = get_tool_model
		else: raise ValueError(f'Unknown LLM mode: {self.mode}')

		ai_response = m()(user_message=self.user_input,
						  conversation_history=self.conversation_history,
						  level=self.level)
		self.result.emit(ai_response)

		# TODO: This is a temporary variable, should be taken from config
		with_feedback = random() > 0.5
		logging.getLogger('llmaker').debug(msg=f'UIInputProcessor.run {with_feedback=}')

		to_process, additional_data = compute_level_diffs(level=self.level)
		progress_delta = int((1 / (1 + (1 if with_feedback else 0) + len(to_process))) * 100)
		
		self.progress_n += progress_delta
		self.progress.emit(self.progress_n)

		if with_feedback:
			# TODO: Message is temporary, should be defined elsewhere
			side_response = m()(user_message='Let\'s CHAT. Tell me what we could change next in the level. Keep your suggestion brief.',
								conversation_history=[],
								level=self.level)
			self.result.emit(side_response)
			
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
			QMessageBox.critical(None, "Error", str(e))