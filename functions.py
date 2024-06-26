import json
from typing import List

from gptfunctionutil import AILibFunction, GPTFunctionLibrary, LibParam, LibParamSpec

from configs import config
from level import Corridor, DIRECTIONS, Encounter, Enemy, Level, OPPOSITE_DIRECTIONS, Room, Trap, Treasure


class DungeonCrawlerFunctions(GPTFunctionLibrary):
	
	def try_call_func(self,
	                  func_name: str,
	                  func_args: str,
	                  level: Level) -> str:
		func_args = json.loads(func_args)
		try:
			operation_result = self.call_by_dict({
				'name':      func_name,
				'arguments': {
					'level': level,
					**func_args
				}
			})
			return operation_result
		except Exception as e:
			return str(e)
	
	@AILibFunction(name='create_room', description='Create a room in the level.',
	               required=['name', 'description', 'room_from'])
	@LibParam(name='The room name')
	@LibParam(description='The room physical characteristics')
	@LibParam(
		room_from='The room the new room connects from. If unspecified, set it to "" if there is no current room, otherwise set it to the current room.')
	def create_room(self, level: Level,
	                name: str,
	                description: str,
	                room_from: str) -> str:
		assert name not in level.rooms.keys(), f'Could not add {name} to the level: {name} already exists.'
		if level.current_room == '':
			assert room_from == '', f'Could not add {name} to the level: room_from must not be set if there is no current room.'
		if level.current_room != '':
			assert room_from != '', f'Could not add {name} to the level: room_from must be set if there exists a current room (current room is {level.current_room}).'
		if room_from != '':
			is_corridor = level.current_room.split('-')
			if len(is_corridor) == 2:
				raise AssertionError(
					f'Could not add {name} to the level: Cannot add a room from a corridor, try adding the room from either {is_corridor[0]} or {is_corridor[1]}.')
			assert room_from in level.rooms.keys(), f'{room_from} is not a valid room name.'
		if room_from != '':
			# try add corridor to connecting room
			n = 0
			for corridor in level.corridors:
				if corridor.room_from == room_from or corridor.room_to == room_from:
					n += 1
			# can only add corridor if the connecting room has at most 3 corridors already
			assert n < 4, f'Could not add {name} to the level: {room_from} has too many connections.'
			# add the new room to the level
			level.rooms[name] = Room(name=name, description=description)
			level.current_room = name
			level.corridors.append(Corridor(room_from=room_from, room_to=name,
			                                name=f'{room_from}-{name}'))
			level.level_geometry[name] = {direction: '' for direction in DIRECTIONS}
			for direction in DIRECTIONS:
				if level.level_geometry[room_from][direction] == '':
					level.level_geometry[room_from][direction] = name
					level.level_geometry[name][OPPOSITE_DIRECTIONS[direction]] = room_from
					break
			return f'Added {name} to the level.'
		else:
			# add the new room to the level
			level.rooms[name] = Room(name=name, description=description)
			level.current_room = name
			level.level_geometry[name] = {direction: '' for direction in DIRECTIONS}
			return f'Added {name} to the level.'
	
	@AILibFunction(name='remove_room', description='Remove the room from the level', required=['name'])
	@LibParam(name='The room name')
	def remove_room(self, level: Level,
	                name: str) -> str:
		assert name in level.rooms.keys(), f'Could not remove {name}: {name} is not in the level.'
		# remove room
		del level.rooms[name]
		del level.level_geometry[name]
		# remove connections from-to deleted room
		to_remove = []
		for i, corridor in enumerate(level.corridors):
			if corridor.room_from == name or corridor.room_to == name:
				to_remove.append(i)
		for i in reversed(to_remove):
			level.corridors.pop(i)
		for other_room_name in level.level_geometry.keys():
			for direction in DIRECTIONS:
				if level.level_geometry[other_room_name][direction] == name:
					level.level_geometry[other_room_name][direction] = ''
		level.current_room = list(level.rooms.keys())[0] if len(level.rooms) > 0 else ''
		return f'{name} has been removed from the dungeon.'
	
	@AILibFunction(name='update_room', description='Update the room',
	               required=['room_reference_name', 'name', 'description'])
	@LibParam(room_reference_name='The original room name')
	@LibParam(name='The room name')
	@LibParam(description='The room physical characteristics')
	def update_room(self, level: Level,
	                room_reference_name: str,
	                name: str,
	                description: str) -> str:
		assert room_reference_name in level.rooms.keys(), f'Could not update {room_reference_name}: {room_reference_name} is not in the level.'
		if name != room_reference_name:
			assert name not in level.rooms.keys(), f'Could not update {room_reference_name}: {name} already exists in the level.'
		# get the current room
		room = level.rooms[room_reference_name]
		# remove it from the list of rooms (since room name can change)
		del level.rooms[room_reference_name]
		# update the room
		room.name = name
		# different description -> sprite must be regenerated
		if room.description != description:
			room.sprite = None
			# entities in the room may be updated, so reset their sprites as well
			for k in room.encounter.entities.keys():
				for entity in room.encounter.entities[k]:
					entity.sprite = None
			# reset the corridor(s) as well
			for corridor in level.corridors:
				if corridor.room_from == room_reference_name:
					corridor.room_from = room.name
					corridor.name = f'{room.name}-{corridor.room_to}'
					corridor.sprite = None
				if corridor.room_to == room_reference_name:
					corridor.room_to = room.name
					corridor.name = f'{corridor.room_from}-{room.name}'
					corridor.sprite = None
		room.description = description
		# add room back
		level.rooms[name] = room
		# update level geometry
		room_connections = level.level_geometry[room_reference_name]
		del level.level_geometry[room_reference_name]
		level.level_geometry[name] = room_connections
		for other_room_name in level.level_geometry.keys():
			for direction in DIRECTIONS:
				if level.level_geometry[other_room_name][direction] == room_reference_name:
					level.level_geometry[other_room_name][direction] = name
		if level.current_room == room_reference_name:
			level.current_room = name
		return f'Updated {room_reference_name}.'
	
	@AILibFunction(name='add_corridor', description='Add a corridor',
	               required=['room_from_name', 'room_to_name', 'corridor_length'])
	@LibParam(room_from_name='The starting room name')
	@LibParam(room_to_name='The connecting room name')
	@LibParamSpec(name='corridor_length', description='The corridor length', minimum=config.dungeon.corridor_min_length,
	              maximum=config.dungeon.corridor_max_length)
	def add_corridor(self, level: Level,
	                 room_from_name: str,
	                 room_to_name: str,
	                 corridor_length: int) -> str:
		assert room_from_name in level.rooms.keys(), f'Room {room_from_name} is not in the level.'
		assert room_to_name in level.rooms.keys(), f'Room {room_to_name} is not in the level.'
		corridor = level.get_corridor(room_from_name, room_to_name, ordered=False)
		assert corridor is None, f'Could not add corridor: a corridor between {room_from_name} and {room_to_name} already exists.'
		assert config.dungeon.corridor_min_length < corridor_length < config.dungeon.corridor_max_length, f'Could not add corridor: corridor_length should be between {config.dungeon.corridor_min_length} and {config.dungeon.corridor_max_length}, not {corridor_length}'
		n = (0, 0)  # number of corridors for each room
		for corridor in level.corridors:
			# count corridors from each room
			if corridor.room_from == room_from_name or corridor.room_to == room_from_name:
				n[0] += 1
			if corridor.room_from == room_to_name or corridor.room_to == room_to_name:
				n[1] += 1
		# only add corridor if each room has at most 3 corridors
		assert n[0] < 4, f'Could not add corridor: {room_from_name} has already 4 connections.'
		assert n[1] < 4, f'Could not add corridor: {room_to_name} has already 4 connections.'
		connectable = (False, None)
		for direction in DIRECTIONS:
			if level.level_geometry[room_from_name][direction] == '' and level.level_geometry[room_to_name][
				OPPOSITE_DIRECTIONS[direction]] == '':
				connectable = (True, direction)
				break
		assert connectable[0], f'Could not add corridor: no direction available for a corridor to be created.'
		direction = connectable[1]
		level.level_geometry[room_from_name][direction] = room_to_name
		level.level_geometry[room_to_name][OPPOSITE_DIRECTIONS[direction]] = room_from_name
		level.corridors.append(Corridor(room_from=room_from_name, room_to=room_to_name,
		                                name=f'{room_from_name}-{room_to_name}',
		                                length=corridor_length,
		                                encounters=[Encounter() for _ in range(corridor_length)]))
		level.current_room = f'{room_from_name}-{room_to_name}'
		return f'Added corridor between {room_from_name} and {room_to_name}.'
	
	@AILibFunction(name='remove_corridor', description='Remove a corridor',
	               required=['room_from_name', 'room_to_name'])
	@LibParam(room_from_name='The starting room name')
	@LibParam(room_to_name='The connecting room name')
	def remove_corridor(self, level: Level,
	                    room_from_name: str,
	                    room_to_name: str) -> str:
		corridor = level.get_corridor(room_from_name, room_to_name, ordered=False)
		assert corridor is not None, f'Corridor between {room_from_name} and {room_to_name} does not exist.'
		# remove the corridor from the level
		level.corridors.remove(corridor)
		for direction in DIRECTIONS:
			for room_a, room_b in [(room_from_name, room_to_name), (room_to_name, room_from_name)]:
				if level.level_geometry[room_a][direction] == room_b:
					level.level_geometry[room_a][direction] = ''
		# update the current room if necessary
		if level.current_room == corridor.name:
			level.current_room = corridor.room_from
		return f'Removed corridor between {room_from_name} and {room_to_name}.'
	
	@AILibFunction(name='update_corridor', description='Update a corridor',
	               required=['room_from_name', 'room_to_name', 'corridor_length'])
	@LibParam(room_from_name='The starting room name')
	@LibParam(room_to_name='The connecting room name')
	@LibParamSpec(name='corridor_length', description='The corridor length', minimum=config.dungeon.corridor_min_length,
	              maximum=config.dungeon.corridor_max_length)
	def update_corridor(self, level: Level,
	                    room_from_name: str,
	                    room_to_name: str,
	                    corridor_length: int) -> str:
		assert config.dungeon.corridor_min_length < corridor_length < config.dungeon.corridor_max_length, f'Could not add corridor: corridor_length should be between {config.dungeon.corridor_min_length} and {config.dungeon.corridor_max_length}, not {corridor_length}'
		corridor = level.get_corridor(room_from_name, room_to_name, ordered=False)
		assert corridor is not None, f'Corridor between {room_from_name} and {room_to_name} does not exist.'
		corridor.length = corridor_length
		# drop encounters if the corridor has shrunk
		if len(corridor.encounters) > corridor.length:
			corridor.encounters = corridor.encounters[:corridor.length]
		# changing the size of the corridor means we need to recreate the background
		corridor.sprite = None
		level.current_room = corridor.name
		return f'Updated corridor between {room_from_name} and {room_to_name}.'
	
	@AILibFunction(name='add_enemies',
	               description='Add enemies to a room or corridor.',
	               required=['room_name', 'cell_index', 'names', 'descriptions', 'species', 'hps', 'dodges', 'prots',
	                         'spds'])
	@LibParam(room_name='The room (or corridor) name')
	@LibParam(
		cell_index='The corridor cell. Set to -1 when targeting a room, otherwise set to a value between 1 and the length of the corridor.')
	@LibParamSpec(name='names', description='The unique name of each enemy', uniqueItems=True)
	@LibParamSpec(name='descriptions', description='The unique physical characteristics of each enemy',
	              uniqueItems=True)
	@LibParam(species='The species of each enemy')
	@LibParamSpec(name='hps',
	              description=f'The health points of each enemy, each value must be between {config.dungeon.min_hp} and  {config.dungeon.max_hp}.')
	@LibParamSpec(name='dodges',
	              description=f'The dodge points of each enemy, each value must be between {config.dungeon.min_dodge} and  {config.dungeon.max_dodge}.')
	@LibParamSpec(name='prots',
	              description=f'The protection points of each enemy, each value must be between {config.dungeon.min_prot} and  {config.dungeon.max_prot}.')
	@LibParamSpec(name='spds',
	              description=f'The speed points of each enemy, each value must be between {config.dungeon.min_spd} and  {config.dungeon.max_spd}.')
	def add_enemies(self, level: Level,
	                room_name: str,
	                names: List[str],
	                descriptions: List[str],
	                species: List[str],
	                hps: List[int],
	                dodges: List[int],
	                prots: List[float],
	                spds: List[int],
	                cell_index: int) -> str:
		assert len(names) == len(descriptions), f'Different number of names and descriptions provided.'
		assert len(names) == len(species), f'Different number of names and species provided.'
		assert len(names) == len(hps), f'Different number of names and hps provided.'
		assert len(names) == len(dodges), f'Different number of names and dodges provided.'
		assert len(names) == len(prots), f'Different number of names and prots provided.'
		assert len(names) == len(spds), f'Different number of names and spds provided.'
		for hp in hps:
			assert config.dungeon.min_hp <= hp <= config.dungeon.max_hp, f'Invalid hp value: {hp}; should be between {config.dungeon.min_hp} and  {config.dungeon.max_hp}.'
		for dodge in dodges:
			assert config.dungeon.min_dodge <= dodge <= config.dungeon.max_dodge, f'Invalid dodge value: {dodge}; should be between {config.dungeon.min_dodge} and  {config.dungeon.max_dodge}.'
		for prot in prots:
			assert config.dungeon.min_prot <= prot <= config.dungeon.max_prot, f'Invalid prot value: {prot}; should be between {config.dungeon.min_prot} and  {config.dungeon.max_prot}.'
		for spd in spds:
			assert config.dungeon.min_spd <= spd <= config.dungeon.max_spd, f'Invalid spd value: {spd}; should be between {config.dungeon.min_spd} and  {config.dungeon.max_spd}.'
		if len(room_name.split('-')) == 2:
			corridor = level.get_corridor(*room_name.split('-'), ordered=False)
			assert corridor is not None, f'Corridor {room_name} does not exist.'
			assert cell_index > 0, f'{room_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
			encounter = corridor.encounters[cell_index - 1]
		else:
			room = level.rooms.get(room_name, None)
			assert room is not None, f'Room {room_name} does not exist.'
			encounter = room.encounter
		assert len(encounter.entities.get('enemy', [])) + len(
			names) <= config.dungeon.max_enemies_per_encounter, f'Could not add enemies: there are already {len(encounter.entities.get("enemy", []))} in {room_name}{" in cell " + str(cell_index + 1) if cell_index != -1 else ""}.'
		added, not_added = [], []
		for i in range(len(names)):
			enemy = Enemy(name=names[i], description=descriptions[i],
			              species=species[i], hp=hps[i], dodge=dodges[i], prot=prots[i], spd=spds[i])
			if encounter.try_add_entity(entity=enemy):
				added.append(f'{names[i]}')
			else:
				not_added.append(f'{names[i]}')
		msg = ''
		if len(added) > 0: msg += f'Added {"; ".join(added)}.'
		if len(not_added) > 0: msg += f'Could not add {"; ".join(not_added)}.'
		return msg
	
	@AILibFunction(name='add_treasure', description='Add a treasure to a room or corridor',
	               required=['room_name', 'cell_index', 'name', 'description', 'loot'])
	@LibParam(room_name='The room (or corridor) name')
	@LibParam(
		cell_index='The corridor cell. Set to -1 when targeting a room, otherwise set to a value between 1 and the length of the corridor.')
	@LibParam(name='The name of the treasures')
	@LibParam(description='The physical characteristics of the treasure')
	@LibParam(loot='The description of the loot in the treasure')
	def add_treasure(self, level: Level,
	                 room_name: str,
	                 name: str,
	                 description: str,
	                 loot: str,
	                 cell_index: int) -> str:
		if len(room_name.split('-')) == 2:
			corridor = level.get_corridor(*room_name.split('-'), ordered=False)
			assert corridor is not None, f'Corridor {room_name} does not exist.'
			assert cell_index > 0, f'{room_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
			encounter = corridor.encounters[cell_index - 1]
		else:
			room = level.rooms.get(room_name, None)
			assert room is not None, f'Room {room_name} does not exist.'
			encounter = room.encounter
		assert len(encounter.entities.get('treasure',
		                                  [])) == 0, f'Could not add treasure: there is already {encounter.entities.get("treasure", [])[0].name} in {room_name}{" in cell " + str(cell_index + 1) if cell_index != -1 else ""}.'
		treasure = Treasure(name=name, description=description, loot=loot)
		return f'Added {name} in {room_name}{" in cell " + str(cell_index + 1) if cell_index != -1 else ""}.' if encounter.try_add_entity(
			treasure) else f'Could not add {name} in {room_name}{" in cell " + str(cell_index + 1) if cell_index != -1 else ""}.'
	
	@AILibFunction(name='add_trap', description='Add a trap to a corridor cell. Traps can be added only to corridors, not to rooms.',
	               required=['corridor_name', 'cell_index', 'name', 'description', 'effect', 'cell_index'])
	@LibParam(corridor_name='The corridor name')
	@LibParam(cell_index='The corridor cell. Set to a value between 1 and the length of the corridor.')
	@LibParam(name='The name of the trap')
	@LibParam(description='The physical characteristics of the trap')
	@LibParam(effect='The effect of the trap')
	def add_trap(self, level: Level,
	             corridor_name: str,
	             name: str,
	             description: str,
	             effect: str,
	             cell_index: int) -> str:
		assert len(corridor_name.split('-')) == 2, f'Traps can only be added only to corridors, but {corridor_name} seems to be a room.'
		corridor = level.get_corridor(*corridor_name.split('-'), ordered=False)
		assert corridor is not None, f'Corridor {corridor_name} does not exist.'
		assert cell_index > 0, f'{corridor_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
		encounter = corridor.encounters[cell_index - 1]
		assert len(encounter.entities.get('trap',
		                                  [])) == 0, f'Could not add trap: there is already {encounter.entities.get("trap", [])[0].name} in {corridor_name} in cell {cell_index}.'
		trap = Trap(name=name, description=description, effect=effect)
		return f'Added {name} in {corridor_name} in cell {cell_index}.' if encounter.try_add_entity(
			trap) else f'Could not add {name} in {corridor_name} in cell {cell_index}.'
	
	# TODO: Updating names of entities should be checked against existing names of entities in the same room!
	
	@AILibFunction(name='update_enemies_properties',
	               description="Update properties of enemies in a room or corridor. Pass the current properties if they're not being updated.",
	               required=['room_name', 'cell_index', 'reference_names', 'names', 'descriptions', 'species', 'hps', 'dodges', 'prots', 'spds'])
	@LibParam(room_name='The room (or corridor) name')
	@LibParam(
		cell_index='The corridor cell. Set to -1 when targeting a room, otherwise set to a value between 1 and the length of the corridor.')
	@LibParam(reference_names='The reference names of each enemy to update')
	@LibParam(names='The unique updated names of each enemy')
	@LibParam(descriptions='The unique updated physical characteristics of each enemy')
	@LibParam(species='The updated species of each enemy')
	@LibParamSpec(name='hps',
	              description=f'The health points of each enemy, each value must be between {config.dungeon.min_hp} and {config.dungeon.max_hp}.')
	@LibParamSpec(name='dodges',
	              description=f'The dodge points of each enemy, each value must be between {config.dungeon.min_dodge} and {config.dungeon.max_dodge}.')
	@LibParamSpec(name='prots',
	              description=f'The protection points of each enemy, each value must be between {config.dungeon.min_prot} and {config.dungeon.max_prot}.')
	@LibParamSpec(name='spds',
	              description=f'The speed points of each enemy, each value must be between {config.dungeon.min_spd} and {config.dungeon.max_spd}.')
	def update_enemies_properties(self, level: Level,
	                              room_name: str,
	                              reference_names: List[str],
	                              names: List[str],
	                              descriptions: List[str],
	                              species: List[str],
	                              hps: List[int],
	                              dodges: List[int],
	                              prots: List[float],
	                              spds: List[int],
	                              cell_index: int) -> str:
		assert len(reference_names) == len(names), f'Different number of reference_names and names provided.'
		assert len(reference_names) == len(
			descriptions), f'Different number of reference_names and descriptions provided.'
		assert len(reference_names) == len(species), f'Different number of reference_names and species provided.'
		assert len(reference_names) == len(hps), f'Different number of reference_names and hps provided.'
		assert len(reference_names) == len(dodges), f'Different number of reference_names and dodges provided.'
		assert len(reference_names) == len(prots), f'Different number of reference_names and prots provided.'
		assert len(reference_names) == len(spds), f'Different number of reference_names and spds provided.'
		for hp in hps:
			assert config.dungeon.min_hp <= hp <= config.dungeon.max_hp, f'Invalid hp value: {hp}; should be between {config.dungeon.min_hp} and  {config.dungeon.max_hp}.'
		for dodge in dodges:
			assert config.dungeon.min_dodge <= dodge <= config.dungeon.max_dodge, f'Invalid dodge value: {dodge}; should be between {config.dungeon.min_dodge} and  {config.dungeon.max_dodge}.'
		for prot in prots:
			assert config.dungeon.min_prot <= prot <= config.dungeon.max_prot, f'Invalid prot value: {prot}; should be between {config.dungeon.min_prot} and  {config.dungeon.max_prot}.'
		for spd in spds:
			assert config.dungeon.min_spd <= spd <= config.dungeon.max_spd, f'Invalid spd value: {spd}; should be between {config.dungeon.min_spd} and  {config.dungeon.max_spd}.'
		if len(room_name.split('-')) == 2:
			corridor = level.get_corridor(*room_name.split('-'), ordered=False)
			assert corridor is not None, f'Corridor {room_name} does not exist.'
			assert cell_index > 0, f'{room_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
			encounter = corridor.encounters[cell_index - 1]
		else:
			room = level.rooms.get(room_name, None)
			assert room is not None, f'Room {room_name} does not exist.'
			encounter = room.encounter
		updated, not_updated = [], []
		for i in range(len(reference_names)):
			enemy = Enemy(name=names[i], description=descriptions[i],
			              species=species[i], hp=hps[i], dodge=dodges[i], prot=prots[i], spd=spds[i])
			if encounter.try_update_entity(entity_reference_name=reference_names[i],
			                               entity_reference_type='enemy',
			                               updated_entity=enemy):
				updated.append(f'{reference_names[i]} is now {str(enemy)}')
			else:
				not_updated.append(reference_names[i])
		msg = ''
		if len(updated) > 0: msg += f'Updated {", ".join(updated)}.'
		if len(not_updated) > 0: msg += f'Could not update {", ".join(not_updated)}.'
		return msg
	
	@AILibFunction(name='update_treasure_properties',
	               description="Update properties of a treasure in a room or corridor. Pass the current properties if they're not being updated.",
	               required=['room_name', 'cell_index', 'reference_name', 'name', 'description', 'loot'])
	@LibParam(room_name='The room (or corridor) name')
	@LibParam(
		cell_index='The corridor cell. Set to None when targeting a room, otherwise set to a value between 1 and the length of the corridor.')
	@LibParam(reference_name='The reference name of the treasure to update')
	@LibParam(name='The updated name of the treasure')
	@LibParam(description='The updated physical characteristics of the treasure')
	@LibParam(loot='The updated loot description of the treasure')
	def update_treasures_properties(self, level: Level,
	                                room_name: str,
	                                reference_name: str,
	                                name: str,
	                                description: str,
	                                loot: str,
	                                cell_index: int) -> str:
		if len(room_name.split('-')) == 2:
			corridor = level.get_corridor(*room_name.split('-'), ordered=False)
			assert corridor is not None, f'Corridor {room_name} does not exist.'
			assert cell_index > 0, f'{room_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
			encounter = corridor.encounters[cell_index - 1]
		else:
			room = level.rooms.get(room_name, None)
			assert room is not None, f'Room {room_name} does not exist.'
			encounter = room.encounter
		treasure = Treasure(name=name, description=description, loot=loot)
		return f'Updated {reference_name}.' if encounter.try_update_entity(entity_reference_name=reference_name,
		                                                                   entity_reference_type='treasure',
		                                                                   updated_entity=treasure) else f'Could not update {reference_name}.'
	
	@AILibFunction(name='update_trap_properties',
	               description="Update properties of a trap in a corridor. Pass the current properties if they're not being updated.",
	               required=['corridor_name', 'cell_index', 'reference_name', 'name', 'description', 'effect'])
	@LibParam(corridor_name='The corridor name')
	@LibParam(
		cell_index='The corridor cell. Set to None when targeting a room, otherwise set to a value between 1 and the length of the corridor.')
	@LibParam(reference_name='The reference name of the trap to update')
	@LibParam(name='The updated name of the traps')
	@LibParam(description='The updated physical characteristics of the trap')
	@LibParam(effect='The updated effects descriptions of the trap')
	def update_trap_properties(self, level: Level,
	                           corridor_name: str,
	                           reference_name: str,
	                           name: str,
	                           description: str,
	                           effect: str,
	                           cell_index: int = None) -> str:
		corridor = level.get_corridor(*corridor_name.split('-'), ordered=False)
		assert corridor is not None, f'Corridor {corridor_name} does not exist.'
		assert cell_index > 0, f'{corridor_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
		encounter = corridor.encounters[cell_index - 1]
		trap = Trap(name=name, description=description, effect=effect)
		return f'Updated {reference_name}.' if encounter.try_update_entity(entity_reference_name=reference_name,
		                                                                   entity_reference_type='trap',
		                                                                   updated_entity=trap) else f'Could not update {reference_name}.'
	
	@AILibFunction(name='remove_entities',
	               description="Remove entities from a room or corridor",
	               required=['room_name', 'cell_index', 'entities_name', 'entities_type'])
	@LibParam(room_name='The room (or corridor) name')
	@LibParam(
		cell_index='The corridor cell. Set to -1 when targeting a room, otherwise set to a value between 1 and the length of the corridor.')
	@LibParam(entities_name='The names of the entities')
	@LibParam(
		entities_type='The type of the entities; must be a list containing only elements with values equal to "enemy", "trap", or "treasure"')
	def remove_entities(self, level: Level,
	                    room_name: str,
	                    entities_name: List[str],
	                    entities_type: List[str],
	                    cell_index: int) -> str:
		assert len(entities_name) == len(entities_type), 'Different number of entities_name and entities_type provided.'
		# ensure entities type is passed correctly
		invalid_entities_type = list(set(entities_type).difference({"enemy", "trap", "treasure"}))
		assert len(invalid_entities_type) == 0, f'Invalid entity type provided: {", ".join(invalid_entities_type)}.'
		if len(room_name.split('-')) == 2:
			corridor = level.get_corridor(*room_name.split('-'), ordered=False)
			assert corridor is not None, f'Corridor {room_name} does not exist.'
			assert cell_index > 0, f'{room_name} is a corridor, but cell_index={cell_index} is invalid, it should be a value between 1 and {corridor.length}.'
			encounter = corridor.encounters[cell_index - 1]
		else:
			room = level.rooms.get(room_name, None)
			assert room is not None, f'Room {room_name} does not exist.'
			encounter = room.encounter
		removed, not_removed = [], []
		for entity_name, entities_type in zip(entities_name, entities_type):
			if encounter.try_remove_entity(entity_name, entities_type):
				removed.append(entity_name)
			else:
				not_removed.append(entity_name)
		msg = ''
		if len(removed) > 0: msg += f'Removed {", ".join(removed)}.'
		if len(not_removed) > 0: msg += f'Could not remove {", ".join(not_removed)}.'
		return msg
