from typing import List, Optional

from chat_message import AnimatedChatMessage, ChatMessage, Conversation

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget


class ConversationWidget(QWidget):
    def __init__(self, parent):
        super(ConversationWidget, self).__init__(parent)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.central_widget = QWidget(self)
        self.central_widget.setProperty("conversation", "yes")
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

        self.placeholder = QLabel(
            parent=self.central_widget, text="Start designing by sending a message!"
        )
        self.placeholder.setProperty("messageType", "placeholder")
        self.central_layout.addWidget(
            self.placeholder, stretch=1, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.animated_message = None
        self.animated_message_timer = None

    def reset(self):
        self.remove_animated_message()
        for message in self.messages:
            self.central_layout.removeWidget(message)
            message.deleteLater()
        self.messages.clear()
        self.conversation = Conversation()
        self.update()

    def add_animated_message(self, operation: str):
        message = AnimatedChatMessage(role="placeholder", msg=operation, interval=500)
        self.animated_message = QLabel(parent=self.central_widget, text=message.content)
        self.animated_message.setTextFormat(Qt.TextFormat.MarkdownText)
        self.animated_message.setProperty("messageType", message.role)
        self.animated_message.setWordWrap(True)
        self.animated_message.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        # Set maximum width to prevent horizontal stretching

        self.animated_message.setMaximumWidth(int(self.width()))

        self.animated_message_timer = QTimer(self)
        self.animated_message_timer.timeout.connect(
            lambda: self._animate_message(message)
        )
        self.animated_message_timer.start(message.interval)

        self.central_layout.addWidget(self.animated_message)

        self.scroll_to_bottom()

    def _animate_message(self, message: AnimatedChatMessage):
        self.animated_message.setText(message.animate())

    def remove_animated_message(self):
        if self.animated_message is not None:
            self.central_layout.removeWidget(self.animated_message)
            self.animated_message.deleteLater()
            self.animated_message = None
        if self.animated_message_timer is not None:
            self.animated_message_timer.stop()
            self.animated_message_timer.deleteLater()
            self.animated_message_timer = None

    def add_message(self, message: str, role: Optional[str] = None):
        role = (
            role
            if role is not None
            else "me" if len(self.conversation) % 2 == 0 else "them"
        )
        new_chat_message = ChatMessage(role=role, msg=message)
        self.conversation.append(new_chat_message)

        new_message = QLabel(parent=self.central_widget, text=new_chat_message.content)
        new_message.setTextFormat(Qt.TextFormat.MarkdownText)
        new_message.setProperty("messageType", new_chat_message.role)
        new_message.setWordWrap(True)
        new_message.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        new_message.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred
        )
        # Set maximum width to prevent horizontal stretching

        new_message.setMaximumWidth(int(self.width()))
        self.messages.append(new_message)

        if self.placeholder.isVisible():
            self.placeholder.hide()
        self.central_layout.addWidget(new_message)

        self.scroll_to_bottom()

    def resizeEvent(self, event):
        # Adjust maximum width of all messages on resize

        for message in self.messages:
            message.setMaximumWidth(self.width())
        super().resizeEvent(event)

    def get_conversation(self) -> List[ChatMessage]:
        return [msg for msg in self.conversation.messages]

    def scroll_to_bottom(self):
        # Scroll to the bottom of the scroll area

        QTimer.singleShot(
            15,  # Allow time for the layout to update before scrolling
            lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ),
        )
