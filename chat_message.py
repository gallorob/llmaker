from typing import List
import json


class ChatMessage:
    def __init__(self,
                 role: str,
                 msg: str):
        self.role = role
        self.msg = msg

    @property
    def content(self) -> str:
        return self.msg

    def str(self) -> str:
        return f"{self.role}: {self.msg}"


class Conversation:
    def __init__(self):
        self.messages: List[ChatMessage] = []
    
    def __len__(self) -> int:
        return len(self.messages)

    def append(self, message: ChatMessage) -> None:
        self.messages.append(message)

    def to_json(self) -> str:
        return json.dumps([message.__dict__ for message in self.messages])
    
    @staticmethod
    def from_json(json_str: str) -> 'Conversation':
        conversation = Conversation()
        for message in json.loads(json_str):
            conversation.messages.append(ChatMessage(**message))
        return conversation