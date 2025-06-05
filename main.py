import logging
import os
import shutil
import sys
from datetime import datetime

import freyr_llm
import tool_llm
from configs import config
from dungeon_despair.domain.configs import config as domain_config
from dungeon_despair.domain.level import Level
from PyQt6.QtWidgets import QApplication, QMessageBox
from ui.main_window import get_splash_screen, MainWindow
from utils import check_server_connection


if __name__ == "__main__":
    domain_config.temp_dir = "./test_results/"

    # clear tmp folder

    if os.path.exists(config.temp_dir):
        shutil.rmtree(config.temp_dir)
    # create tmp folder if it does not exist

    if not os.path.exists(config.temp_dir):
        os.makedirs(config.temp_dir)
    # create log folder if it does not exist

    if not os.path.exists("./logs"):
        os.makedirs("./logs")
    # create levels folder if it does not exist

    if not os.path.exists(config.levels_dir):
        os.makedirs(config.levels_dir)
    # create scenarios folder if it does not exist

    if not os.path.exists(config.scenarios_dir):
        os.makedirs(config.scenarios_dir)
    log_filename = f'./logs/log_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
        )
    )
    logging.getLogger("llmaker").setLevel(logging.DEBUG)
    logging.getLogger("llmaker").addHandler(handler)
    logging.getLogger("gui").setLevel(logging.DEBUG)
    logging.getLogger("gui").addHandler(handler)

    app = QApplication(sys.argv)

    splash_screen = get_splash_screen()
    splash_screen.show()

    splash_screen.showMessage("Checking server connection...")
    if not check_server_connection():
        QMessageBox.critical(
            None, 'Connection Error', f"Server at {config.server_ip}:{config.server_port} is unreachable or down!"
        )
        sys.exit(-1)

    splash_screen.showMessage("Loading Large Language Models...")
    if config.llm_mode == "freyr":
        freyr_llm.load_local_llm(splash_screen)
    else:
        tool_llm.load_local_llm(splash_screen)
    splash_screen.showMessage("Loaded Large Language Models")
    win = MainWindow(level=Level())

    win.show()
    win.update()

    splash_screen.finish(win)

    sys.exit(app.exec())
