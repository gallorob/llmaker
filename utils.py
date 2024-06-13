from typing import List, Union

from level import Corridor, Enemy, Entity, Room, Trap, Treasure


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