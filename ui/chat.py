from typing import List

from PyQt6.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QLabel, QSizePolicy


class ConversationWidget(QWidget):
	def __init__(self, parent):
		super(ConversationWidget, self).__init__(parent)
		
		self.scroll_area = QScrollArea(self)
		self.scroll_area.setWidgetResizable(True)
		
		self.central_widget = QWidget(self)
		self.central_widget.setProperty('conversation', 'yes')
		self.scroll_area.setWidget(self.central_widget)
		
		self.central_layout = QVBoxLayout(self.central_widget)
		self.central_layout.setSpacing(0)
		self.central_layout.setContentsMargins(0, 0, 0, 0)
		
		# Hacky way to avoid resizing messages when they are too few
		# Praise https://stackoverflow.com/questions/63438039/qt-dont-stretch-widgets-in-qvboxlayout
		self.central_layout.addStretch()
		
		main_layout = QVBoxLayout(self)
		main_layout.addWidget(self.scroll_area)
		self.setLayout(main_layout)
		
		self.messages: List[QLabel] = []
	
	def reset(self):
		for message in self.messages:
			self.central_layout.removeWidget(message)
			message.deleteLater()
		self.messages.clear()
		self.update()
	
	def load_conversation(self, conversation):
		for i, line in enumerate(conversation.split('\n')):
			line = line.replace('You: ', '').replace('AI: ', '')
			self.add_message(line)
	
	def add_message(self, message):
		new_message = QLabel(parent=self.central_widget, text=message)
		new_message.setProperty('messageType', 'me' if len(self.messages) % 2 == 0 else 'them')
		
		new_message.setWordWrap(True)
		new_message.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
		
		# Set maximum width to prevent horizontal stretching
		new_message.setMaximumWidth(int(self.width()))
		
		self.messages.append(new_message)
		self.central_layout.addWidget(new_message)
		
		self.update()
	
	def resizeEvent(self, event):
		# Adjust maximum width of all messages on resize
		for message in self.messages:
			message.setMaximumWidth(self.width())
		super().resizeEvent(event)
	
	def get_conversation(self):
		return '\n'.join([message.text() for message in self.messages])
