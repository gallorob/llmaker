from enum import Enum, auto
from typing import List, Union, Any, Tuple

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.entities.trap import Trap
from dungeon_despair.domain.entities.treasure import Treasure
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room


class ToolMode(Enum):
	USER = auto()
	LLM = auto()


class ThemeMode(Enum):
	LIGHT = auto()
	DARK = auto()
	
	
def basic_room_description(room: Room) -> str:
	return f'<h2>{room.name}</h2><h3><i>{room.description}</i></h3>'


def basic_corridor_description(corridor: Corridor) -> str:
	return f'<h3>Corridor between <i>{corridor.room_from}</i> and <i>{corridor.room_to}</i></h3>'


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


def clear_strings_for_prompt(strings: List[str]):
	return [s.lower().replace('.', '').strip() for s in strings]


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
	for corridor in level.corridors:
		if corridor.sprite is None:
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
