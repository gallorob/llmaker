from typing import List

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QBrush, QColor, QPixmap, QPainter
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QVBoxLayout, QGraphicsPixmapItem

from configs import config
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from dungeon_despair.domain.utils import derive_rooms_from_corridor_name, is_corridor
from utils import ThemeMode, rich_entity_description


# TODO: Clicking on an entity shows up a more detailed dialog for that enemy


class EncounterPreviewWidget(QWidget):
	def __init__(self, parent, level: Level):
		super(EncounterPreviewWidget, self).__init__(parent)
		
		self.level = level
		
		self.scene = QGraphicsScene(self)
		self.view = QGraphicsView(self.scene)
		self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
		
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
				background_image = QPixmap(room.sprite)
			else:
				room = self.level.get_corridor(*derive_rooms_from_corridor_name(self.level.current_room), ordered=True)
				corridor_chunks = [QPixmap(sprite) for sprite in room.sprites]
				background_image = QPixmap(corridor_chunks[0].width() * len(corridor_chunks), corridor_chunks[0].height())
				painter = QPainter(background_image)
				for i, chunk in enumerate(corridor_chunks):
					painter.drawPixmap(QRectF(i * chunk.width(), 0,
					                          chunk.width(), chunk.height()),
					                   chunk,
					                   QRectF(0, 0, chunk.width(), chunk.height()))
				painter.end()
				
			
			self.scene.setSceneRect(0, 0, background_image.width(), background_image.height())
			item = self.scene.addPixmap(background_image)
			item.setPos(0, 0)
			self.view.fitInView(item, Qt.AspectRatioMode.KeepAspectRatioByExpanding)
			
			if is_corridor(self.level.current_room):
				self.view.horizontalScrollBar().setValue(0)
			
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
