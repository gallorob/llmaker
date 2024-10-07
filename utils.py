import logging
from enum import Enum, auto
from typing import List, Union, Any, Tuple, Dict

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.entities.trap import Trap
from dungeon_despair.domain.entities.treasure import Treasure
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from sd_backend import generate_room, generate_corridor, generate_entity


class ToolMode(Enum):
	USER = 'user'
	LLM = 'llm'


class ThemeMode(Enum):
	LIGHT = 'light'
	DARK = 'dark'
	
	
def basic_room_description(room: Room) -> str:
	return f'<h2>{room.name}</h2><h3><i>{room.description}</i></h3>'


def basic_corridor_description(corridor: Corridor) -> str:
	return f'<h3>Corridor between <i>{corridor.room_from}</i> and <i>{corridor.room_to}</i></h3>'


def basic_entity_description(entity: Entity) -> str:
	description = f'<h1>{entity.name}</h1>'
	description += f'<h4>{entity.description}</h4>'
	
	if isinstance(entity, Enemy):
		description += f'<h6>HP: {entity.hp}; DODGE: {entity.dodge}; PROT: {entity.prot:.2f}; SPD: {entity.spd}</h6>'
	elif isinstance(entity, Treasure):
		description += f'<h6>Loot: {entity.loot}</h6>'
	elif isinstance(entity, Trap):
		description += f'<h6>Effect: {entity.effect}</h6>'
	
	return description

def rich_entity_description(entity: Entity) -> str:
	rich_description = f'<h1>{entity.name}</h1>'
	rich_description += f'<h4>{entity.description}</h4>'
	
	if isinstance(entity, Enemy):
		rich_description += f'<h6>Species: {entity.species}</h6>'
		rich_description += f'<h6>HP: {entity.hp}</h6>'
		rich_description += f'<h6>DODGE: {entity.dodge}</h6>'
		rich_description += f'<h6>PROT: {entity.prot:.2f}</h6>'
		rich_description += f'<h6>SPD: {entity.spd}</h6>'
	elif isinstance(entity, Treasure):
		rich_description += f'<h6>Loot: {entity.loot}</h6>'
	elif isinstance(entity, Trap):
		rich_description += f'<h6>Effect: {entity.effect}</h6>'
	
	return rich_description


def compute_level_diffs(level: Level) -> Tuple[List[Union[Room, Corridor, Entity]], List[Any]]:
	to_process: List[Union[Room, Corridor, Entity]] = []
	additional_data: List[Any] = []
	
	for room_name in level.rooms.keys():
		room = level.rooms[room_name]
		if room.sprite is None:
			to_process.append(room)
			additional_data.append(None)
		for entity_type in room.encounter.entities:
			for entity in room.encounter.entities[entity_type]:
				if entity.sprite is None:
					to_process.append(entity)
					additional_data.append(
						{'entity_type': entity_type, 'room_name': room.name, 'room_description': room.description})
	for corridor in level.corridors.values():
		if len(corridor.sprites) == 0 or None in corridor.sprites:
			to_process.append(corridor)
			room_from, room_to = level.rooms[corridor.room_from], level.rooms[corridor.room_to]
			additional_data.append({'room_descriptions': [room_from.description, room_to.description]})
		for i, encounter in enumerate(corridor.encounters):
			for entity_type in encounter.entities:
				for entity in encounter.entities[entity_type]:
					if entity.sprite is None:
						to_process.append(entity)
						room_from = level.rooms[corridor.room_from]
						additional_data.append(
							{'entity_type': entity_type, 'room_name': room_from.name,
							 'room_description': room_from.description})
	
	return to_process, additional_data

def process_diff(obj: Any,
                 additional_data: Dict[str, str]) -> None:
	if isinstance(obj, Room):
		logging.info(f'Room {obj.name} has no sprite; generating...')
		obj.sprite = generate_room(room_name=obj.name,
		                           room_description=obj.description)
	elif isinstance(obj, Corridor):
		logging.info(f'Room {obj.name} has no sprite; generating...')
		obj_data = additional_data
		obj.sprites = generate_corridor(room_names=[obj.room_from, obj.room_to],
		                               corridor_length=obj.length + 2,
		                               **obj_data,
		                               corridor_sprites=obj.sprites)
	elif isinstance(obj, Entity):
		obj_data = additional_data
		logging.info(f'Entity {obj.name} has no sprite; generating...')
		obj.sprite = generate_entity(entity_name=obj.name,
		                             entity_description=obj.description,
		                             **obj_data)
	else:
		raise ValueError(f'Unsupported object type: {type(obj)}')