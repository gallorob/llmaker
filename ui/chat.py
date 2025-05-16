from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QScrollArea, QVBoxLayout, QLabel, QSizePolicy

from chat_message import ChatMessage, Conversation

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
		self.conversation: Conversation = Conversation()

		self.placeholder = QLabel(parent=self.central_widget, text="Start designing by sending a message!")
		self.placeholder.setProperty('messageType', 'placeholder')
		self.central_layout.addWidget(self.placeholder, stretch=1, alignment=Qt.AlignmentFlag.AlignCenter)
	
	def reset(self):
		for message in self.messages:
			self.central_layout.removeWidget(message)
			message.deleteLater()
		self.messages.clear()
		self.conversation = Conversation()
		self.update()
		
	def add_message(self, message: str, role: Optional[str] = None):
		role = role if role is not None else 'me' if len(self.conversation) % 2 == 0 else 'them'
		new_chat_message = ChatMessage(role=role,
								       msg=message)
		self.conversation.append(new_chat_message)

		new_message = QLabel(parent=self.central_widget, text=new_chat_message.content)
		new_message.setProperty('messageType', new_chat_message.role)
		new_message.setWordWrap(True)
		new_message.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
		# Set maximum width to prevent horizontal stretching
		new_message.setMaximumWidth(int(self.width()))
		self.messages.append(new_message)

		if self.placeholder.isVisible():
			self.placeholder.hide()
		self.central_layout.addWidget(new_message)
		
		self.update()
	
	def resizeEvent(self, event):
		# Adjust maximum width of all messages on resize
		for message in self.messages:
			message.setMaximumWidth(self.width())
		super().resizeEvent(event)
	
	def get_conversation(self) -> List[str]:
		return [message.content for message in self.conversation.messages]

	def update(self):
		# Scroll to the bottom of the scroll area
		self.scroll_area.verticalScrollBar().setValue(self.scroll_area.verticalScrollBar().maximum())
		super().update()