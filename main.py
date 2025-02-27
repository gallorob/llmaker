import logging
import os
import shutil
import sys
from datetime import datetime

from PyQt6.QtWidgets import QApplication

from dungeon_despair.domain.configs import config as domain_config

from configs import config
from dungeon_despair.domain.level import Level
from freyr_llm import load_local_llm

from sd_backend import load_stablediff_models
from ui.ui_elements import MainWindow, get_splash_screen

STARTING_LEVEL = 'empty'
# STARTING_LEVEL = 'from_file'
# FILE_NAME = './my_levels/baselevel.bin'
FILE_NAME = './my_levels/t5_with_attacks.bin'

logging.getLogger('llmaker').setLevel(logging.DEBUG)


if __name__ == '__main__':
	domain_config.temp_dir = './test_results/'
	logging.getLogger('llmaker').setLevel(logging.DEBUG)
	log_filename = f'./logs/log_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
	handler = logging.FileHandler(log_filename)
	handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
	logging.getLogger('llmaker').addHandler(handler)
	
	app = QApplication(sys.argv)
	
	splash_screen = get_splash_screen()
	splash_screen.show()
	
	# clear tmp folder
	if os.path.exists(config.temp_dir):
		shutil.rmtree(config.temp_dir)
	# create tmp folder if it does not exist
	if not os.path.exists(config.temp_dir):
		os.makedirs(config.temp_dir)
	
	logging.getLogger('llmaker').info('Started loading large language models...')
	splash_screen.showMessage('Loading Large Language Models...')
	load_local_llm(splash_screen)
	splash_screen.showMessage('Loaded Large Language Models')
	logging.getLogger('llmaker').info('Started loading diffusion models...')
	splash_screen.showMessage('Loading Stable Diffusion Models...')
	load_stablediff_models(splash_screen)
	splash_screen.showMessage('Loaded Stable Diffusion Models')
	logging.getLogger('llmaker').info('Diffusion models loaded')
	
	win = MainWindow(level=Level())
	
	if STARTING_LEVEL == 'from_file' and os.path.exists(FILE_NAME):
		logging.getLogger('llmaker').info(f'Loading level from file: {FILE_NAME}')
		splash_screen.showMessage('Loading level from file...')
		level, _ = Level.load_from_file(FILE_NAME)
		win.set_level(level)
		with open('tmp_conversation.txt', 'r') as f:
			conversation = f.read()
		win.chat_area.load_conversation(conversation)
		logging.getLogger('llmaker').info('Updating GUI...')
		win.map_preview.show_map_preview()
		win.room_preview.show_room_preview()
		splash_screen.showMessage('Loaded level from file')
		logging.getLogger('llmaker').info('Loaded level from file')
	elif STARTING_LEVEL == 'empty':
		logging.getLogger('llmaker').info('No initial level set')
	else:
		raise ValueError(f'STARTING_LEVEL {STARTING_LEVEL} is not valid')

	win.show()
	win.update()
	
	splash_screen.finish(win)
	
	sys.exit(app.exec())
