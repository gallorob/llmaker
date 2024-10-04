import logging
import time
from typing import List, Union, Any

from PyQt6.QtCore import QObject, pyqtSignal

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from llm_backend import chat_llm
from sd_backend import generate_room, generate_entity, generate_corridor
from utils import compute_level_diffs


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
		# to_process: List[Union[Room, Corridor, Entity]] = []
		# additional_data: List[Any] = []
		#
		# for room_name in self.level.rooms.keys():
		# 	room = self.level.rooms[room_name]
		# 	if room.sprite is None:
		# 		to_process.append(room)
		# 		additional_data.append(None)
		# 	for entity_type in room.encounter.entities:
		# 		for entity in room.encounter.entities[entity_type]:
		# 			if entity.sprite is None:
		# 				to_process.append(entity)
		# 				additional_data.append(
		# 					{'entity_type': entity_type, 'room_name': room.name, 'room_description': room.description})
		# for corridor in self.level.corridors:
		# 	if corridor.sprite is None:
		# 		to_process.append(corridor)
		# 		room_from, room_to = self.level.rooms[corridor.room_from], self.level.rooms[corridor.room_to]
		# 		additional_data.append({'room_descriptions': [room_from.description, room_to.description]})
		# 	for i, encounter in enumerate(corridor.encounters):
		# 		for entity_type in encounter.entities:
		# 			for entity in encounter.entities[entity_type]:
		# 				if entity.sprite is None:
		# 					to_process.append(entity)
		# 					room_from = self.level.rooms[corridor.room_from]
		# 					additional_data.append(
		# 						{'entity_type': entity_type, 'room_name': room_from.name,
		# 						 'room_description': room_from.description})
		
		to_process, additional_data = compute_level_diffs(level=self.level)
		
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
				                               **obj_data,
				                               corridor_sprites=obj.sprites)
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