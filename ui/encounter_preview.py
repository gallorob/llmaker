import os
from functools import partial
from typing import List

from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QBrush, QColor, QPixmap, QPainter, QMouseEvent
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QVBoxLayout, QGraphicsPixmapItem
from dungeon_despair.domain.utils import ModifierType, get_enum_by_value

from configs import config
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from ui.dyn_dialog import EnemyPreviewDialog
from utils import ThemeMode, get_modifier_icon, basic_entity_description


def show_enemy_dialog(event: QMouseEvent, parent: QWidget, enemy: Enemy):
        if event.button() == Qt.MouseButton.LeftButton:
            dialog = EnemyPreviewDialog(enemy=enemy, parent=parent)
            dialog.exec()


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
			if self.level.current_room in self.level.rooms.keys():
				room = self.level.rooms[self.level.current_room]
				background_image = QPixmap(os.path.join(config.room.save_dir, room.sprite))
			else:
				room = self.level.corridors[self.level.current_room]
				corridor_chunks = [QPixmap(os.path.join(config.corridor.save_dir, sprite)) for sprite in room.sprites]
				background_image = QPixmap(corridor_chunks[0].width() * len(corridor_chunks), corridor_chunks[0].height())
				painter = QPainter(background_image)
				for i, chunk in enumerate(corridor_chunks):
					painter.drawPixmap(QRectF(i * chunk.width(), 0,
					                          chunk.width(), chunk.height()),
					                   chunk,
					                   QRectF(0, 0, chunk.width(), chunk.height()))
				painter.end()
				
			
			self.scene.setSceneRect(0, 0, background_image.width(), background_image.height())
			self.scene.addPixmap(background_image)

			view_rect = self.view.viewport().rect()
			image_rect = background_image.rect()

			if self.level.current_room in self.level.rooms.keys():
				scale_x = view_rect.width() / image_rect.width()
			else:
				c = self.level.corridors[self.level.current_room]
				scale_x = view_rect.width() / (image_rect.width() / c.length)
			scale_y = view_rect.height() / image_rect.height()

			self.view.resetTransform()
			self.view.scale(scale_x, scale_y)
			
			w, h = background_image.width(), background_image.height()

			def __draw_entities(entities: List[Entity], x_offset, y_offset, scaled_entity_width) -> None:
				for i in range(config.dungeon.max_enemies_per_encounter):
					if i < len(entities):
						entity = entities[i]
						entity_sprite = QPixmap(os.path.join(config.entity.save_dir, entity.sprite))
						entity_rect = QGraphicsPixmapItem(entity_sprite)
						entity_rect.setScale(config.ui.entity_scale)
						entity_rect.setToolTip(basic_entity_description(entity=entity))
						entity_rect.setPos(x_offset + scaled_entity_width * i,
						                   y_offset - (entity_sprite.height() * entity_rect.scale()))
						if isinstance(entity, Enemy):
							entity_rect.mousePressEvent = partial(show_enemy_dialog, enemy=entity, parent=self)
						self.scene.addItem(entity_rect)

						if hasattr(entity, 'modifier'):
							modifier = entity.modifier
							if modifier is not None:
								modifier_sprite = QPixmap(get_modifier_icon(get_enum_by_value(ModifierType, modifier.type)))
								modifier_rect = QGraphicsPixmapItem(modifier_sprite)
								modifier_rect.setScale(config.ui.modifier_scale)
								modifier_rect.setToolTip(str(modifier))
								scaled_modifier_width = config.ui.modifier_scale * modifier_sprite.width()
								offset_mod = (scaled_entity_width - scaled_modifier_width) / 2
								modifier_rect.setPos(x_offset + (scaled_entity_width * i) + offset_mod,
													y_offset)
								self.scene.addItem(modifier_rect)

			
			scaled_entity_width = config.entity.width * config.ui.entity_scale
			y_offset = 5 * h / 6
			if isinstance(room, Room):
				for entities in [room.encounter.treasures, room.encounter.enemies]:
					total_width = scaled_entity_width * len(entities)
					x_offset = (w - total_width) / 2
					__draw_entities(entities, x_offset, y_offset, scaled_entity_width)
			else:
				for i, encounter in enumerate(room.encounters):
					for entities in [encounter.treasures, encounter.traps, encounter.enemies]:
						total_width = scaled_entity_width * len(entities)
						x_offset = ((i + 1) * (w / (room.length + 2)) + (w / (room.length + 2) / 2)) - (total_width / 2)
						__draw_entities(entities, x_offset, y_offset, scaled_entity_width)
