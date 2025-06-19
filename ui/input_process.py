import logging
import sys
import time
from typing import Any, Dict, List

from autodef import AutoFeedback
from chat_message import ChatMessage

from dungeon_despair.domain.level import Level
from freyr_llm import get_freyr_model
from gptfunctionutil import LibCommand

from PyQt6.QtCore import pyqtSignal, pyqtSlot, QObject, QRunnable
from tool_llm import get_tool_model
from utils import compute_level_diffs, LLMMode, process_diff


class WorkerSignals(QObject):
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


class UIInputProcessor(QRunnable):
    def __init__(
        self,
        level: Level,
        user_input: str,
        conversation_history: List[ChatMessage],
        llm_mode: LLMMode,
    ):
        super().__init__()
        self.signals = WorkerSignals()
        self.level = level
        self.user_input = user_input
        self.conversation_history = conversation_history
        self.mode = llm_mode
        if self.mode == LLMMode.FREYR:
            self.model = get_freyr_model()
            self.feedback = AutoFeedback(model=get_freyr_model)
        elif self.mode == LLMMode.TOOL:
            self.model = get_tool_model()
        else:
            raise ValueError(f"Unknown LLM mode: {self.mode}")

    @pyqtSlot()
    def run(self):
        try:
            self.progress_n = 0

            ai_response = self.model(
                user_message=self.user_input,
                conversation_history=self.conversation_history,
                level=self.level,
            )
            self.signals.result.emit(ai_response)

            to_process, additional_data = compute_level_diffs(level=self.level)

            with_feedback = (
                len(to_process) > 0 and self.feedback and self.feedback.enabled
            )

            progress_delta = int(
                (1 / (1 + (1 if with_feedback else 0) + len(to_process))) * 100
            )

            self.progress_n += progress_delta
            self.signals.progress.emit(self.progress_n)

            if with_feedback:
                logging.getLogger("llmaker").debug(
                    msg="Proactive feedback is enabled, processing additional feedback."
                )
                feedback_response = self.feedback.give_uninformed_feedback(
                    level=self.level, conversation_history=self.conversation_history
                )
                self.signals.result.emit(feedback_response)

                self.progress_n += progress_delta
                self.signals.progress.emit(self.progress_n)
            for i, obj in enumerate(to_process):
                process_diff(obj, additional_data[i])
                self.progress_n += progress_delta
                self.signals.progress.emit(self.progress_n)
            time.sleep(0.5)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value))
        finally:
            self.signals.finished.emit()


class DebugInputProcessor(QRunnable):
    def __init__(self, kwargs: Dict[str, Any], func: LibCommand):
        super(DebugInputProcessor, self).__init__()
        self.kwargs = kwargs
        self.func = func
        self.signals = WorkerSignals()

    @pyqtSlot()
    def run(self):
        self.progress_n = 0
        try:
            submission_output = self.func.command(**self.kwargs)

            to_process, additional_data = compute_level_diffs(
                level=self.kwargs["level"]
            )

            progress_delta = int((1 / (1 + len(to_process))) * 100)

            self.progress_n += progress_delta
            self.signals.progress.emit(self.progress_n)

            for i, obj in enumerate(to_process):
                process_diff(obj, additional_data[i])
                self.progress_n += progress_delta
                self.signals.progress.emit(self.progress_n)
            time.sleep(0.5)
            self.signals.result.emit(submission_output)
        except Exception:
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value))
        finally:
            self.signals.finished.emit()
