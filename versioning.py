from copy import deepcopy
from typing import List, Optional, Tuple

from chat_message import ChatMessage
from dungeon_despair.domain.level import Level


class VersionHandler:
    def __init__(self, level: Level, chat: List[Optional[ChatMessage]]):
        self.__levels = [deepcopy(level)]
        self.__chat = deepcopy(chat)
        self.__idx = 0
        self.__chat_to_idx = {self.__idx: len(self.__chat)}

    def __dereferenced_level_version(self, idx: int) -> Level:
        return deepcopy(self.__levels[idx])

    def __get_chat(self, idx: int) -> List[Optional[ChatMessage]]:
        return deepcopy(self.__chat[: self.__chat_to_idx[idx]])

    @property
    def get_current_level(self) -> Level:
        return self.__dereferenced_level_version(self.__idx)

    @property
    def can_undo(self) -> bool:
        return self.__idx > 0

    @property
    def can_redo(self) -> bool:
        return self.__idx < len(self.__levels) - 1

    def commit(self, level: Level, chat: List[Optional[ChatMessage]]):
        self.__levels = self.__levels[: self.__idx + 1]
        self.__levels.append(deepcopy(level))
        self.__chat.extend(deepcopy(chat[self.__chat_to_idx[self.__idx] :]))
        self.__idx += 1
        self.__chat_to_idx[self.__idx] = len(self.__chat)

    def undo(self) -> Tuple[Level, List[Optional[ChatMessage]]]:
        if self.__idx == 0:
            return self.__dereferenced_level_version(self.__idx), self.__get_chat(
                self.__idx
            )
        self.__idx -= 1
        return self.__dereferenced_level_version(self.__idx), self.__get_chat(
            self.__idx
        )

    def redo(self) -> Tuple[Level, List[Optional[ChatMessage]]]:
        if self.__idx == len(self.__levels) - 1:
            return self.__dereferenced_level_version(self.__idx), self.__get_chat(
                self.__idx
            )
        self.__idx += 1
        return self.__dereferenced_level_version(self.__idx), self.__get_chat(
            self.__idx
        )
