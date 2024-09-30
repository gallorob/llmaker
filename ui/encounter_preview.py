from typing import List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QPixmap
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QVBoxLayout, QGraphicsPixmapItem

from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from dungeon_despair.domain.utils import derive_rooms_from_corridor_name, is_corridor
from level import Entity
from utils import ThemeMode, rich_entity_description


class EncounterPreviewWidget(QWidget):
	def __init__(self, parent, level: Level):
		super(EncounterPreviewWidget, self).__init__(parent)
		
		self.level = level
		
		self.scene = QGraphicsScene(self)
		self.view = QGraphicsView(self.scene)
		
		self.view_layout = QVBoxLayout(self)
		self.view_layout.addWidget(self.view)
	
	def paintEvent(self, a0):
		self.scene.clear()
		self.show_room_preview()
	
	def show_room_preview(self):
		self.scene.setBackgroundBrush(
			QBrush(QColor('#1e1d23' if self.parent().parent().parent().theme == ThemeMode.DARK else '#ececec')))
		
		if self.level.current_room != '':
			if not is_corridor(self.level.current_room):
				room = self.level.rooms[self.level.current_room]
			else:
				room = self.level.get_corridor(*derive_rooms_from_corridor_name(self.level.current_room), ordered=True)
			
			# background_image = QPixmap.fromImage(ImageQt(room.sprite.convert("RGBA")))
			background_image = QPixmap(room.sprite)
			self.scene.setSceneRect(0, 0, background_image.width(), background_image.height())
			item = self.scene.addPixmap(background_image)
			item.setPos(0, 0)
			self.view.fitInView(item, Qt.AspectRatioMode.KeepAspectRatio)
			
			w, h = self.scene.width(), self.scene.height()
			
			def __draw_entities(entities: List[Entity], x_offset, y_offset, scaled_entity_width) -> None:
				for i in range(config.dungeon.max_enemies_per_encounter):
					if i < len(entities):
						entity = entities[i]
						# entity_sprite = QPixmap.fromImage(ImageQt(entity.sprite.convert("RGBA")))
						entity_sprite = QPixmap(entity.sprite)
						entity_rect = QGraphicsPixmapItem(entity_sprite)
						entity_rect.setScale(config.ui.entity_scale)
						entity_rect.setToolTip(rich_entity_description(entity=entity))
						entity_rect.setPos(x_offset + scaled_entity_width * i,
						                   y_offset - (entity_sprite.height() * entity_rect.scale()))
						self.scene.addItem(entity_rect)
			
			if isinstance(room, Room):
				scaled_entity_width = config.entity.width * config.ui.entity_scale
				y_offset = 5 * h / 6
				x_offset = w / 2 - scaled_entity_width / 2
				
				__draw_entities(room.encounter.entities['treasure'], x_offset, y_offset, scaled_entity_width)
				
				enemies = room.encounter.entities['enemy']
				if len(enemies) > 1:
					x_offset -= (scaled_entity_width * (len(enemies) - 1)) / 2
				__draw_entities(enemies, x_offset, y_offset, scaled_entity_width)
			else:
				scaled_entity_width = config.entity.width * config.ui.entity_scale
				y_offset = 0.9 * h
				for i, encounter in enumerate(room.encounters):
					x_offset = ((i + 1) * w / (room.length + 2)) + w / (room.length + 2) / 2 - (scaled_entity_width / 2)
					
					__draw_entities(encounter.entities['treasure'], x_offset, y_offset, scaled_entity_width)
					__draw_entities(encounter.entities['trap'], x_offset, y_offset, scaled_entity_width)
					
					enemies = encounter.entities['enemy']
					if len(enemies) > 1:
						x_offset -= (scaled_entity_width * (len(enemies) - 1)) / 2
					__draw_entities(enemies, x_offset, y_offset, scaled_entity_width)
