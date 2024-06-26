import os
import pickle
from enum import Enum
from typing import Dict, List, Optional, Tuple

import PIL.Image
from pydantic import BaseModel, Field

from configs import config

DIRECTIONS = ['LEFT', 'UP', 'RIGHT', 'DOWN']
OPPOSITE_DIRECTIONS = {
	'UP':    'DOWN',
	'DOWN':  'UP',
	'LEFT':  'RIGHT',
	'RIGHT': 'LEFT'
}


class Entity(BaseModel):
	class Config:
		arbitrary_types_allowed = True
	
	name: str = Field(..., description="The name of the entity.", required=True)
	description: str = Field(..., description="The description of the entity.", required=True)
	sprite: str = Field(default=None, description='The sprite for the entity.', required=False, exclude=True)
	
	def __str__(self):
		return f'Name={self.name} Description="{self.description}"'


class Enemy(Entity):
	species: str = Field(..., description="The enemy species.", required=True)
	hp: int = Field(..., description="The enemy HP.", required=True)
	dodge: int = Field(..., description="The enemy dodge stat.", required=True)
	prot: float = Field(..., description="The enemy prot stat.", required=True)
	spd: int = Field(..., description="The enemy spd stat.", required=True)
	
	def __str__(self):
		return f'{super().__str__()} Species={self.species} HP={self.hp} DODGE={self.dodge} PROT={self.prot} SPD={self.spd}'


class Trap(Entity):
	effect: str = Field(..., description="The effect of the trap.", required=True)
	
	def __str__(self):
		return f'{super().__str__()} Effect={self.effect}'


class Treasure(Entity):
	loot: str = Field(..., description="The loot in the treasure.", required=True)
	
	def __str__(self):
		return f'{super().__str__()} Loot={self.loot}'


class EntityClass(Enum):
	ENEMY = Enemy
	TRAP = Trap
	TREASURE = Treasure


entityclass_to_str = {
	Enemy:    'enemy',
	Trap:     'trap',
	Treasure: 'treasure'
}

entityclass_thresolds = {
	Enemy:    config.dungeon.max_enemies_per_encounter,
	Trap:     config.dungeon.max_traps_per_encounter,
	Treasure: config.dungeon.max_treasures_per_encounter
}


class Encounter(BaseModel):
	class Config:
		arbitrary_types_allowed = True
	
	entities: Dict[str, List[Entity]] = Field(default={k: [] for k in entityclass_to_str.values()},
	                                          description="The entities for this encounter.", required=True)
	
	def __str__(self):
		s = ''
		for k in self.entities.keys():
			all_type_str = [str(x) for x in self.entities[k]]
			unique_with_count = [f'{all_type_str.count(x)}x {x}' for x in all_type_str]
			s += f'\n\t\t{str(k).lower()}: {", ".join(unique_with_count)}'
		return s
	
	def try_add_entity(self, entity: Entity) -> bool:
		klass = entityclass_to_str[entity.__class__]
		if klass not in self.entities.keys(): self.entities[klass] = []
		if len(self.entities[klass]) < entityclass_thresolds[entity.__class__]:
			# add the entity
			self.entities[klass].append(entity)
			return True
		return False
	
	def try_remove_entity(self, entity_name: str, entity_type: str) -> bool:
		n = None
		for i, entity in enumerate(self.entities[entity_type]):
			if entity.name == entity_name:
				n = i
				break
		if n is not None:
			self.entities[entity_type].pop(n)
			return True
		return False
	
	def try_update_entity(self, entity_reference_name: str, entity_reference_type: str, updated_entity: Entity) -> bool:
		for i, entity in enumerate(self.entities[entity_reference_type]):
			if entity.name == entity_reference_name:
				if updated_entity.description == self.entities[entity_reference_type][i].description:
					updated_entity.sprite = self.entities[entity_reference_type][i].sprite
				self.entities[entity_reference_type][i] = updated_entity
				return True
		return False


class Room(BaseModel):
	class Config:
		arbitrary_types_allowed = True
	
	name: str = Field(..., description="The name of the room.", required=True)
	description: str = Field(..., description="The description of the room", required=True)
	encounter: Encounter = Field(default=Encounter(), description='The encounter in the room.', required=True)
	sprite: str = Field(default=None, description='The sprite for the room.', required=False, exclude=True)
	
	def __str__(self):
		return f'{self.name}: {self.description};{self.encounter}'


class Corridor(BaseModel):
	class Config:
		arbitrary_types_allowed = True
	
	room_from: str = Field(..., description="The room the corridor is connected from.", required=True)
	room_to: str = Field(..., description="The room the corridor is connects to.", required=True)
	name: str = Field('', description='The name of the corridor.', required=True)
	length: int = Field(default=config.dungeon.corridor_min_length, description="The length of the corridor",
	                    required=True)
	encounters: List[Encounter] = Field(default=[Encounter() for _ in range(config.dungeon.corridor_min_length)],
	                                    description="The encounters in the corridor.", required=True)
	sprite: str = Field(default=None, description='The sprite for the corridor.', required=False, exclude=True)
	
	def __str__(self):
		s = f'{self.name}: from {self.room_from} to {self.room_to}, {self.length} cells long;'
		for i, e in enumerate(self.encounters):
			s += f'\n\tCell {i+1} {str(e)}'
		return s


class Level(BaseModel):
	class Config:
		arbitrary_types_allowed = True
	
	rooms: Dict[str, Room] = Field(default={}, description="The rooms in the level.", required=True)
	corridors: List[Corridor] = Field(default=[], description="The corridors in the level.", required=True)
	
	current_room: str = Field(default='', description="The currently selected room.", required=True)
	
	level_geometry: Dict[str, Dict[str, str]] = Field(default={}, description="The geometry of the level.",
	                                                  required=True)
	
	def save_to_file(self, filename: str, conversation: str) -> None:
		# get all images
		# all_images = os.listdir(config.temp_dir)
		all_images = []
		for room in self.rooms.values():
			all_images.append(room.sprite)
			for entity_type in room.encounter.entities.keys():
				for entity in room.encounter.entities[entity_type]:
					all_images.append(entity.sprite)
		for corridor in self.corridors:
			all_images.append(corridor.sprite)
			for encounter in corridor.encounters:
				for entity_type in encounter.entities.keys():
					for entity in encounter.entities[entity_type]:
						all_images.append(entity.sprite)
		images = {image_path: PIL.Image.open(os.path.join(config.temp_dir, image_path)) for image_path in all_images}
		bin_data = {
			'level': self,
			'images': images,
			'conversation': conversation
		}
		with open(filename, 'wb') as f:
			pickle.dump(bin_data, f)
	
	@staticmethod
	def load_from_file(filename: str) -> Tuple["Level", str]:
		with open(filename, 'rb') as f:
			bin_data = pickle.load(f)
			images = bin_data['images']
			for fpath, image in images.items():
				image.save(os.path.join(config.temp_dir, fpath))
			return bin_data['level'], bin_data['conversation']
	
	def __str__(self) -> str:
		# This is the GLOBAL level description
		# TODO: Implement the LOCAL level description that only gives specific information for the current room
		level_description = 'Rooms:\n' + '\n\t'.join([str(self.rooms[k]) for k in self.rooms.keys()]) + '\n'
		level_description += 'Corridors:\n' + '\n\t'.join([str(c) for c in self.corridors]) + '\n'
		level_description += f'Current room: {self.current_room}'
		return level_description
	
	def get_corridor(self, room_from_name, room_to_name, ordered=False) -> Optional[Corridor]:
		for c in self.corridors:
			if (c.room_from == room_from_name and c.room_to == room_to_name) or (
				not ordered and (c.room_from == room_to_name and c.room_to == room_from_name)):
				return c
		return None
