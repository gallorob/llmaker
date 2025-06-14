from random import random
from typing import Callable, List, Optional

from chat_message import ChatMessage
from configs import config, resource_path

from dungeon_despair.domain.level import Level
from freyr_llm import FreyrLLM
from utils import check_applicable_operation


class AutoFeedback:
    def __init__(self, model: Callable[[], Optional[FreyrLLM]]):
        self.model = model

    @property
    def enabled(self) -> bool:
        return random() > config.llm.proactive.chance

    def get_filtered_operations(self, level: Level) -> List[str]:
        all_operations = self.model().tools_as_dict()
        filtered_operations = {}
        for op, description in all_operations.items():
            if check_applicable_operation(op_name=op, level=level):
                filtered_operations[op] = description
        return filtered_operations

    def give_uninformed_feedback(
        self, level: Level, conversation_history: List[ChatMessage]
    ) -> str:
        with open(resource_path(config.llm.proactive.msg), "r") as f:
            side_msg = f.read()
        side_msg = side_msg.format(
            filtered_operations=self.get_filtered_operations(level=level)
        )

        valid_conversation_history = self.model().trim_and_convert_conversation(
            conversation_history
        )

        side_response = self.model().chat(
            conversation_history=valid_conversation_history,
            user_message=side_msg,
            level=level,
        )

        return side_response

    def give_informed_feedback(
        self, level: Level, conversation_history: List[ChatMessage]
    ) -> str:
        raise NotImplementedError("Informed feedback is not implemented yet.")
