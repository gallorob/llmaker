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
from requests.exceptions import ConnectionError
from ui.main_window import get_splash_screen, MainWindow
from utils import check_server_connection


def setup_logging(log_filename):
    handler = logging.FileHandler(log_filename)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s"
        )
    )
    logger = logging.getLogger("llmaker")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)

    gui_logger = logging.getLogger("gui")
    gui_logger.setLevel(logging.DEBUG)
    gui_logger.addHandler(handler)


def setup_directories():
    directories = [config.temp_dir, "./logs", config.levels_dir, config.scenarios_dir]
    for directory in directories:
        if os.path.exists(directory) and directory == config.temp_dir:
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    domain_config.temp_dir = "./test_results/"

    setup_directories()

    log_filename = f'./logs/log_{datetime.now().strftime("%Y%m%d%H%M%S")}.log'
    setup_logging(log_filename)

    logging.getLogger("llmaker").info(
        f"Version: v0.0.9.4-hotfix3; Username: {config.username}; Mode: {config.start_mode}; Can switch mode: {config.can_switch_mode}"
    )

    app = QApplication(sys.argv)
    splash_screen = get_splash_screen()
    splash_screen.show()

    splash_screen.showMessage("Checking server connection...")
    if not check_server_connection():
        QMessageBox.critical(
            None, "Connection Error", f"Server is unreachable or down!"
        )
        sys.exit(-1)
    splash_screen.showMessage("Loading Large Language Models...")
    try:
        if config.llm_mode == "freyr":
            freyr_llm.load_local_llm(splash_screen)
        else:
            tool_llm.load_local_llm(splash_screen)
    except ConnectionError as e:
        logging.error(f"Connection error: {e}")
        QMessageBox.critical(None, "Connection Error", e.args[0])
        sys.exit(-1)
    splash_screen.showMessage("Loaded Large Language Models")
    win = MainWindow(level=Level())
    win.show()
    win.update()

    splash_screen.finish(win)
    sys.exit(app.exec())
