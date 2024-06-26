from datetime import datetime
import logging
import os
import random
import shutil
import sys

from PIL import Image
from PyQt6.QtWidgets import QApplication

from level import Encounter, Enemy, Level
from configs import config
from functions import DungeonCrawlerFunctions
from sd_backend import generate_corridor, generate_entity, generate_room, load_stablediff_models
from ui.ui_elements import MainWindow, get_splash_screen


# STARTING_LEVEL = 'empty'
# STARTING_LEVEL = 'random'
STARTING_LEVEL = 'from_file'


# TODO: Add a bunch more logging everywhere


def get_randomised_level():
	level = Level()
	funcs = DungeonCrawlerFunctions()
	
	rooms_name = ['Forest Clearing', 'Lava Pit', 'Old Cellar', 'Dungeon', 'Ethereal Candyland',
	              'Space Station Control Room', 'Swamp Room']
	rooms_description = [
		'A grassy clearing in a dense forest.', 'A hellish landscape with lava flowing on the floor.',
		'A musty old cellar.', 'An underground dungeon brick room.', 'A candyland by the seaside in the clouds.',
		'A futuristic sci-fi control room in a space station.', 'An underground swamp room.'
	]
	
	enemies = [
		Enemy(name='Red Cap Goblin', description='A goblin wearing a red cap and wielding a rusty sword.',
		      species='Humanoid', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Beast Demon', description='A winged furry demon with a lion head.',
		      species='Beast', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Ghost', description='A ghost wielding a chainsaw.',
		      species='Ethereal', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Menacing Alien', description='A green alien armed with a laser pistol.',
		      species='Alien', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Emaciated Kobold', description='An emaciated Kobold wielding a large wooden stick.',
		      species='Humanoid', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Gingerbread Man', description='A giant gingerbread man wielding a sugar cane.',
		      species='Fantasy', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Werecat', description='A manly man turned into a giant depressed cat.',
		      species='Humanoid/Beast', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Bog Witch', description='A bog witch wearing tattered robes',
		      species='Humanoid', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
		Enemy(name='Swamp Beast', description='A massive beast covered in algae and mud.',
		      species='Mythical', hp=random.randint(1, 20), dodge=random.randint(1, 10),
		      prot=random.random(), spd=random.randint(1, 10)),
	]
	
	order_idxs = [i for i in range(len(rooms_name))]
	connect_idxs = []
	random.shuffle(order_idxs)
	
	n_rooms = random.randint(1, len(order_idxs))
	order_idxs = order_idxs[:n_rooms]
	
	for i in order_idxs:
		room_a_name = rooms_name[i]
		if len(connect_idxs) > 0:
			prev_i = i
			while i == prev_i:
				prev_i = random.sample(connect_idxs, 1)[0]
			prev_room_name = rooms_name[prev_i]
		else:
			prev_room_name = ''
		funcs.create_room(level=level,
		                  name=room_a_name,
		                  description=rooms_description[i],
		                  room_from=prev_room_name)
		
		room_enemies = list(range(len(enemies)))
		random.shuffle(room_enemies)
		n_enemies = random.randint(0, config.dungeon.corridor_max_length)
		room_enemies = room_enemies[:n_enemies]
		for j in room_enemies:
			enemy = enemies[j].model_copy()
			level.rooms[room_a_name].encounter.try_add_entity(enemy)
		
		if len(connect_idxs) > 0:
			corridor = level.get_corridor(room_from_name=room_a_name, room_to_name=prev_room_name)
			corridor.length = random.randint(2, 4)
			corridor.encounters = [Encounter() for _ in range(corridor.length)]
		connect_idxs.append(i)
	
	level.current_room = rooms_name[order_idxs[0]]
	
	for room in level.rooms.values():
		if room.sprite is None:
			if os.path.exists(f'./test_results/{room.name.lower()}.png'):
				room.sprite = Image.open(f'./test_results/{room.name.lower()}.png')
			else:
				logging.info(f'Generating sprite for {room.name}...')
				room.sprite = generate_room(room.name, room.description)
		for enemy in room.encounter.entities['enemy']:
			if enemy.sprite is None:
				if os.path.exists(f'./test_results/{enemy.name.lower()}_{room.name.lower()}.png'):
					enemy.sprite = Image.open(f'./test_results/{enemy.name.lower()}_{room.name.lower()}.png')
				else:
					logging.info(f'Generating sprite for {enemy.name}...')
					enemy.sprite = generate_entity(enemy.name, enemy.description, 'enemy',
					                               room.name, room.description)
	
	for corridor in level.corridors:
		if corridor.sprite is None:
			if os.path.exists(f'./test_results/{corridor.room_from.lower()}-{corridor.room_to.lower()}_{corridor.length + 2}_corridor.png'):
				corridor.sprite = Image.open(f'./test_results/{corridor.room_from.lower()}-{corridor.room_to.lower()}_{corridor.length + 2}_corridor.png')
			else:
				logging.info(f'Generating sprite for {corridor.room_from}-{corridor.room_to}...')
				corridor.sprite = generate_corridor(room_names=[corridor.room_from, corridor.room_to],
				                                    room_descriptions=[level.rooms[corridor.room_from].description, level.rooms[corridor.room_to].description],
				                                    corridor_length=corridor.length + 2)
		for encounter in corridor.encounters:
			for enemy in encounter.entities['enemy']:
				if enemy.sprite is None:
					if os.path.exists(f'./test_results/{enemy.name.lower()}_{corridor.room_from.lower()}.png'):
						enemy.sprite = Image.open(f'./test_results/{enemy.name.lower()}_{corridor.room_from.lower()}.png')
					else:
						logging.info(f'Generating sprite for {enemy.name}...')
						enemy.sprite = generate_entity(enemy.name, enemy.description, 'enemy',
						                               corridor.room_from, level.rooms[corridor.room_from].description)
	
	return level


if __name__ == '__main__':
	logging.getLogger().setLevel(logging.DEBUG)
	log_filename = f'./logs/log_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
	handler = logging.FileHandler(log_filename)
	handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logging.getLogger().addHandler(handler)

	app = QApplication(sys.argv)
	
	splash_screen = get_splash_screen()
	splash_screen.show()
	
	# clear tmp folder
	if os.path.exists(config.temp_dir):
		shutil.rmtree(config.temp_dir)
	# create tmp folder if it does not exist
	if not os.path.exists(config.temp_dir):
		os.makedirs(config.temp_dir)
	
	logging.info('Started loading diffusion models...')
	splash_screen.showMessage('Loading Stable Diffusion Models...')
	load_stablediff_models(splash_screen)
	splash_screen.showMessage('Loaded Stable Diffusion Models')
	logging.info('Diffusion models loaded')
	
	win = MainWindow(level=Level())
	
	if STARTING_LEVEL == 'from_file' and os.path.exists('my_levels/t1.bin'):
		logging.info('Loading level from file: T1.bin')
		splash_screen.showMessage('Loading level from file...')
		# TODO: Regenerate T1 and save it so we can load it whenever later
		level, conversation = Level.load_from_file('my_levels/t1.bin')
		win.set_level(level)
		for i, line in enumerate(conversation.split('\n')):
			win.chat_area.append(f'{"<b>You</b>" if i % 2 == 0 else "<b>AI</b>"}: {line}')
		logging.info('Updating GUI...')
		win.map_preview.show_map_preview()
		win.room_preview.show_room_preview()
		splash_screen.showMessage('Loaded level from file')
		logging.info('Loaded level from file')
	elif STARTING_LEVEL == 'random':
		logging.info('Generating a random new level...')
		splash_screen.showMessage('Generating a random new level...')
		win.set_level(get_randomised_level())
		win.map_preview.show_map_preview()
		win.room_preview.show_room_preview()
		splash_screen.showMessage('New level generated')
		logging.info('New level generated')
	elif STARTING_LEVEL == 'empty':
		logging.info('No initial level set')
	else:
		raise ValueError(f'STARTING_LEVEL {STARTING_LEVEL} is not valid')

	win.update()
	
	win.show()
	
	splash_screen.finish(win)

	sys.exit(app.exec())
