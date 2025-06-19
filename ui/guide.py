import os

from configs import resource_path
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QDialog, QLabel, QScrollArea, QVBoxLayout, QWidget


class GuideDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("LLMaker Guide")
        self.setWindowIcon(QIcon(resource_path("assets/llmaker_logo.png")))
        self.setMinimumSize(QSize(400, 300))

        # Create layout

        layout = QVBoxLayout(self)

        # Create scroll area

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Load HTML content

        guide_path = resource_path("assets/llmaker_guide.html")
        if os.path.exists(guide_path):
            with open(guide_path, "r", encoding="utf-8") as f:
                html_content = f.read()

                # Create label with HTML content

                guide_label = QLabel()
                guide_label.setTextFormat(Qt.TextFormat.RichText)
                guide_label.setOpenExternalLinks(True)
                guide_label.setText(html_content)
                guide_label.setWordWrap(True)
                scroll.setWidget(guide_label)
        else:
            error_label = QLabel("Guide file not found!")
            scroll.setWidget(error_label)
        # Add scroll area to main layout

        layout.addWidget(scroll)
