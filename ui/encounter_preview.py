import copy
import os
from functools import partial
from typing import List

from configs import config, resource_path
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from dungeon_despair.domain.utils import get_enum_by_value, ModifierType

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QVBoxLayout,
    QWidget,
)
from ui.dyn_dialog import EnemyPreviewDialog
from utils import basic_entity_description, get_modifier_icon, ThemeMode


def show_enemy_dialog(event: QMouseEvent, parent: QWidget, enemy: Enemy):
    if event.button() == Qt.MouseButton.LeftButton:
        dialog = EnemyPreviewDialog(enemy=enemy, parent=parent)
        dialog.exec()


class EntityGraphicsItem(QGraphicsPixmapItem):
    def __init__(self, entity: Entity, parent: QWidget):
        super().__init__()
        self.entity = entity
        self.parent = parent

        self.setAcceptHoverEvents(True)
        self.setFlag(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable)

    def mousePressEvent(self, event: QMouseEvent):
        if isinstance(self.entity, Enemy):
            if event.button() == Qt.MouseButton.LeftButton:
                show_enemy_dialog(event, self.parent, self.entity)
                event.accept()  # Important: Accept the event (else the same entity will be selected again)


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

        self.scene.setItemIndexMethod(QGraphicsScene.ItemIndexMethod.BspTreeIndex)
        self.view.setMouseTracking(True)

    def paintEvent(self, a0):
        super().paintEvent(a0)

    def refresh(self):
        self.scene.clear()
        self.show_room_preview()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.rescale_view()

    def rescale_view(self):
        if self.scene.items():
            view_rect = self.view.viewport().rect()
            scene_rect = self.scene.sceneRect()
            if not scene_rect.isNull():
                if self.level.current_room in self.level.rooms.keys():
                    # Scale for rooms

                    scale_x = view_rect.width() / scene_rect.width()
                    scale_y = view_rect.height() / scene_rect.height()
                    self.view.resetTransform()
                    self.view.scale(scale_x, scale_y)
                else:
                    # Do not scale for corridors, keep them scrollable

                    self.view.resetTransform()

    def check_scene(self):
        if self.level.current_room != "":
            curr_area = (
                self.level.rooms[self.level.current_room]
                if self.level.current_room in self.level.rooms
                else self.level.corridors[self.level.current_room]
            )
            if isinstance(curr_area, Room):
                if not curr_area.sprite:
                    self.scene.clear()
            else:
                if None in curr_area.sprites:
                    self.scene.clear()

    def show_room_preview(self):
        if self.level.current_room != "":
            # Clear existing items before adding new ones

            self.scene.clear()
            if self.level.current_room in self.level.rooms.keys():
                room = self.level.rooms[self.level.current_room]
                background_image = QPixmap(
                    os.path.join(config.room.save_dir, room.sprite)
                )
            else:
                room = self.level.corridors[self.level.current_room]
                corridor_chunks = [
                    QPixmap(os.path.join(config.corridor.save_dir, sprite))
                    for sprite in room.sprites
                ]
                background_image = QPixmap(
                    corridor_chunks[0].width() * len(corridor_chunks),
                    corridor_chunks[0].height(),
                )
                painter = QPainter(background_image)
                for i, chunk in enumerate(corridor_chunks):
                    painter.drawPixmap(
                        QRectF(i * chunk.width(), 0, chunk.width(), chunk.height()),
                        chunk,
                        QRectF(0, 0, chunk.width(), chunk.height()),
                    )
                painter.end()
            self.scene.setSceneRect(
                0, 0, background_image.width(), background_image.height()
            )
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

            def __draw_entities(
                entities: List[Entity], x_offset, y_offset, scaled_entity_width
            ) -> None:
                for i in range(config.dungeon.max_enemies_per_encounter):
                    if i < len(entities):
                        entity = entities[i]
                        entity_sprite = QPixmap(
                            os.path.join(config.entity.save_dir, entity.sprite)
                        )
                        entity_rect = EntityGraphicsItem(entity=entity, parent=self)
                        entity_rect.setPixmap(entity_sprite)
                        entity_rect.setScale(config.ui.entity_scale)
                        entity_rect.setToolTip(basic_entity_description(entity=entity))
                        entity_rect.setPos(
                            x_offset + scaled_entity_width * i,
                            y_offset - (entity_sprite.height() * entity_rect.scale()),
                        )
                        entity_rect.setAcceptHoverEvents(True)
                        entity_rect.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
                        # Add to scene with increasing z-value

                        self.scene.addItem(entity_rect)
                        entity_rect.setZValue(i + 1)  # Ensure items don't overlap

                        if hasattr(entity, "modifier"):
                            modifier = entity.modifier
                            if modifier is not None:
                                modifier_sprite = QPixmap(
                                    resource_path(
                                        get_modifier_icon(
                                            get_enum_by_value(
                                                ModifierType, modifier.type
                                            )
                                        )
                                    )
                                )
                                modifier_rect = QGraphicsPixmapItem(modifier_sprite)
                                modifier_rect.setScale(config.ui.modifier_scale)
                                modifier_rect.setToolTip(str(modifier))
                                scaled_modifier_width = (
                                    config.ui.modifier_scale * modifier_sprite.width()
                                )
                                offset_mod = (
                                    scaled_entity_width - scaled_modifier_width
                                ) / 2
                                modifier_rect.setPos(
                                    x_offset + (scaled_entity_width * i) + offset_mod,
                                    y_offset,
                                )
                                self.scene.addItem(modifier_rect)

            scaled_entity_width = config.entity.width * config.ui.entity_scale
            y_offset = 5 * h / 6
            if isinstance(room, Room):
                for entities in [room.encounter.treasures, room.encounter.enemies]:
                    total_width = scaled_entity_width * len(entities)
                    x_offset = (w - total_width) / 2
                    __draw_entities(entities, x_offset, y_offset, scaled_entity_width)
            else:
                scaled_entity_width *= 0.75  # Scale down for corridors
                for i, encounter in enumerate(room.encounters):
                    for entities in [
                        encounter.treasures,
                        encounter.traps,
                        encounter.enemies,
                    ]:
                        total_width = scaled_entity_width * len(entities)
                        x_offset = (
                            (i + 1) * (w / (room.length + 2))
                            + (w / (room.length + 2) / 2)
                        ) - (total_width / 2)
                        __draw_entities(
                            entities, x_offset, y_offset, scaled_entity_width
                        )
