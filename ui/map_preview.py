from functools import partial
from typing import Union, Tuple, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor, QWheelEvent, QDragMoveEvent
from PyQt6.QtWidgets import QWidget, QGraphicsScene, QGraphicsView, QVBoxLayout, QGraphicsRectItem

from configs import config
from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.room import Room
from dungeon_despair.domain.utils import is_corridor, derive_rooms_from_corridor_name, Direction, opposite_direction
from utils import ThemeMode, basic_room_description, basic_corridor_description


class MapPreviewWidget(QWidget):
	def __init__(self, parent, level: Level):
		super(MapPreviewWidget, self).__init__(parent)
		
		self.level = level
		
		self.scene = QGraphicsScene(self)
		self.view = QGraphicsView(self.scene)
		
		self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		
		self.view_layout = QVBoxLayout(self)
		self.view_layout.addWidget(self.view)
		
		self.drawn_rooms = []
		
		self.corridor_draw_size = None
		self.room_draw_size = None
	
	def paintEvent(self, a0):
		self.show_map_preview()
	
	def get_rects(self,
	              room: Union[Room, Corridor],
	              offset_x: float,
	              offset_y: float,
	              direction: Optional[Direction],
	              selected=False) -> Tuple[List[QGraphicsRectItem], int, int]:
		def __update_offsets(offset_x, offset_y, direction, draw_size) -> Tuple[int, int]:
			if direction == Direction.NORTH:
				offset_y -= draw_size
			elif direction == Direction.SOUTH:
				offset_y += draw_size
			elif direction == Direction.WEST:
				offset_x -= draw_size
			elif direction == Direction.EAST:
				offset_x += draw_size
			else:
				pass
			return offset_x, offset_y

		rects = []

		if isinstance(room, Room):
			draw_offset_x, draw_offset_y = offset_x, offset_y
			rect = QGraphicsRectItem(0, 0, self.room_draw_size, self.room_draw_size)
			rect.setBrush(QBrush(QColor(config.ui.selected_color if selected else config.ui.unselected_color)))
			rect.setToolTip(basic_room_description(room))
			draw_offset_x -= self.room_draw_size / 2
			draw_offset_y -= self.room_draw_size / 2
			rect.setPos(draw_offset_x, draw_offset_y)
			rect.mousePressEvent = partial(self.parent().parent().parent().on_room_press, room.name)
			rects.append(rect)
			self.drawn_rooms.append(room.name)
			
			for direction in Direction:
				other_room_name = self.level.connections[room.name][direction]
				if other_room_name != '':
					corridor = self.level.get_corridor(room_from_name=room.name, room_to_name=other_room_name)
					if corridor is not None:
						if f'{corridor.room_from}_{corridor.room_to}' not in self.drawn_rooms:
							next_offset_x, next_offset_y = __update_offsets(offset_x, offset_y, direction,
							                                                self.room_draw_size)
							other_rects, _, _ = self.get_rects(corridor, next_offset_x, next_offset_y, direction)
							rects.extend(other_rects)

		else:
			corridor_offset = (self.room_draw_size - self.corridor_draw_size) / 2
			if direction is not None:
				draw_direction = direction
				draw_offset_x, draw_offset_y = __update_offsets(offset_x, offset_y, opposite_direction[direction],
				                                                corridor_offset)
			else:
				draw_offset_x, draw_offset_y = offset_x, offset_y
				corridor_draw_length_offset = self.corridor_draw_size * room.length / 2
				if self.level.connections[room.room_from][Direction.NORTH] == room.room_to or \
						self.level.connections[room.room_from][Direction.SOUTH] == room.room_to:
					draw_direction = Direction.SOUTH
					draw_offset_y -= corridor_draw_length_offset
				else:
					draw_direction = Direction.EAST
					draw_offset_x -= corridor_draw_length_offset

			draw_offset_x -= self.corridor_draw_size / 2
			draw_offset_y -= self.corridor_draw_size / 2
			for i in range(room.length):
				rect = QGraphicsRectItem(0, 0, self.corridor_draw_size, self.corridor_draw_size)
				rect.setBrush(QBrush(QColor(config.ui.selected_color if selected else config.ui.unselected_color)))
				rect.setToolTip(basic_corridor_description(room))
				rect.setPos(draw_offset_x, draw_offset_y)
				draw_offset_x, draw_offset_y = __update_offsets(draw_offset_x, draw_offset_y, draw_direction,
				                                                self.corridor_draw_size)
				rect.mousePressEvent = partial(self.parent().parent().parent().on_corridor_press, room.room_from,
				                               room.room_to)
				rects.append(rect)
			self.drawn_rooms.append(f'{room.room_from}_{room.room_to}')

			if direction is not None:
				offset_x, offset_y = __update_offsets(offset_x, offset_y, draw_direction,
				                                      room.length * self.corridor_draw_size)
				for other_room in [room.room_from, room.room_to]:
					if other_room not in self.drawn_rooms:
						other_rects, _, _ = self.get_rects(self.level.rooms[other_room], offset_x, offset_y, direction)
						rects.extend(other_rects)
			else:
				offset_x, offset_y = __update_offsets(offset_x, offset_y, draw_direction, corridor_offset)
				for (room_a, room_b) in [(room.room_from, room.room_to), (room.room_to, room.room_from)]:
					if room_a not in self.drawn_rooms:
						for new_direction in Direction:
							if self.level.connections[room_a][new_direction] == room_b:
								if new_direction == draw_direction:
									new_offset_x, new_offset_y = __update_offsets(offset_x, offset_y,
									                                              opposite_direction[new_direction],
									                                              room.length * self.corridor_draw_size / 2 + self.room_draw_size)
								else:
									new_offset_x, new_offset_y = __update_offsets(offset_x, offset_y,
									                                              opposite_direction[new_direction],
									                                              room.length * self.corridor_draw_size / 2)
								other_rects, _, _ = self.get_rects(self.level.rooms[room_a], new_offset_x, new_offset_y,
								                                   new_direction)
								rects.extend(other_rects)

		return rects, offset_x, offset_y
	
	def show_map_preview(self):
		self.drawn_rooms = []
		x, y = self.rect().x(), self.rect().y()
		if self.corridor_draw_size is None:
			self.corridor_draw_size = self.rect().height() * config.ui.minimap_corridor_scale
		if self.room_draw_size is None:
			self.room_draw_size = self.rect().height() * config.ui.minimap_room_scale

		self.scene.clear()
		self.scene.setBackgroundBrush(
			QBrush(QColor('#1e1d23' if self.parent().parent().parent().theme == ThemeMode.DARK else '#ececec')))

		if self.level.current_room != '':
			if not is_corridor(self.level.current_room):
				room = self.level.rooms[self.level.current_room]
			else:
				room = self.level.get_corridor(*derive_rooms_from_corridor_name(self.level.current_room), ordered=True)
			
			rects, _, _ = self.get_rects(room=room,
			                             offset_x=x // 2, offset_y=y // 2,
			                             direction=None, selected=True)
			
			for rect in rects:
				self.scene.addItem(rect)
	
	def wheelEvent(self, a0: QWheelEvent):
		num_degrees = a0.angleDelta().y() / 8
		num_steps = num_degrees / 15
		self.room_draw_size += (config.ui.minimap_zoom_step * config.ui.minimap_room_scale) * num_steps
		self.corridor_draw_size += (config.ui.minimap_zoom_step * config.ui.minimap_corridor_scale) * num_steps
		
		self.update()
