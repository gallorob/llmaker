import base64
import json
import logging
import math
import os
from enum import auto, Enum
from hashlib import sha224
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

from configs import config

from dungeon_despair.domain.corridor import Corridor
from dungeon_despair.domain.entities.enemy import Enemy
from dungeon_despair.domain.entities.entity import Entity
from dungeon_despair.domain.entities.trap import Trap
from dungeon_despair.domain.entities.treasure import Treasure
from dungeon_despair.domain.level import Level
from dungeon_despair.domain.modifier import Modifier
from dungeon_despair.domain.room import Room
from dungeon_despair.domain.utils import ModifierType
from PIL import Image
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QGuiApplication, QPixmap, QScreen
from PyQt6.QtWidgets import QMainWindow
from requests import get, post, Response
from requests.exceptions import ConnectionError


class ToolMode(Enum):
    USER = "user"
    LLM = "llm"


class LLMMode(Enum):
    FREYR = "freyr"
    TOOL = "tool"


class ThemeMode(Enum):
    LIGHT = "light"
    DARK = "dark"


def check_server_connection() -> bool:
    server_url = config.server_url
    try:
        response = get(f"{server_url}/ollama_list_models")
        if response.status_code == 200:
            return True
        return False
    except ConnectionError:
        return False


def send_to_server(data: Optional[Dict[str, Any]], endpoint) -> Response:
    server_url = config.server_url
    if data:
        uid = sha224(config.username.encode("utf-8")).hexdigest()
        payload = {"uid": uid, **data}
        response = post(f"{server_url}/{endpoint}", json=payload)
    else:
        response = get(f"{server_url}/{endpoint}")
    if response.status_code != 200:
        logging.getLogger("llmaker").error(
            f"Server error: {response.status_code} - {response.text}"
        )
        raise ConnectionError(json.loads(response.text)["message"])
    return response.json()


def convert_and_save(b64_img: str, fname: str, dirname: str) -> str:
    full_name = os.path.join(dirname, fname)
    with open(full_name, "wb") as f:
        f.write(base64.b64decode(b64_img))
    return os.path.basename(full_name)


def rgb_sum_app_entropy(window: QMainWindow) -> float:
    screen: QScreen = QGuiApplication.primaryScreen()
    geo: QRect = window.frameGeometry()
    screenshot: QPixmap = screen.grabWindow(
        0, geo.x(), geo.y(), geo.width(), geo.height()
    ).toImage()

    width, height = screenshot.width(), screenshot.height()
    rgb_sum_counts = {}

    for y in range(height):
        for x in range(width):
            rgb = screenshot.pixelColor(x, y)
            s = rgb.red() + rgb.green() + rgb.blue()
            rgb_sum_counts[s] = rgb_sum_counts.get(s, 0) + 1
    total_pixels = width * height
    entropy = 0.0
    for count in rgb_sum_counts.values():
        p = count / total_pixels
        entropy -= p * math.log2(p)
    return entropy


def get_modifier_icon(modifier_type: ModifierType):
    if modifier_type == ModifierType.BLEED:
        return config.icons.bleed
    elif modifier_type == ModifierType.HEAL:
        return config.icons.heal
    elif modifier_type == ModifierType.SCARE:
        return config.icons.scare
    elif modifier_type == ModifierType.STUN:
        return config.icons.stun
    else:
        raise ValueError(f"Unknown modifier type: {modifier_type.value}")


def basic_room_description(room: Room) -> str:
    return f"<h2>{room.name}</h2><h3><i>{room.description}</i></h3>"


def basic_corridor_description(corridor: Corridor) -> str:
    return f"<h3>Corridor between <i>{corridor.room_from}</i> and <i>{corridor.room_to}</i></h3>"


def basic_entity_description(entity: Entity) -> str:
    description = f"<h1>{entity.name}</h1>"
    description += f"<h4>{entity.description}</h4>"

    if isinstance(entity, Enemy):
        description += f"<h6>HP: {entity.hp}; DODGE: {entity.dodge}; PROT: {entity.prot:.2f}; SPD: {entity.spd}</h6><h6>{len(entity.attacks)}/{config.dungeon.max_num_attacks} attacks.</h6>"
    elif isinstance(entity, Treasure):
        description += f"<h6>Loot: {entity.loot}</h6><h6>Trapped Chance: {entity.trapped_chance:.0%} dealing {entity.dmg}DMG</h6>"
    elif isinstance(entity, Trap):
        description += f"<h6>Effect: {entity.effect}</h6><h6>Chance: {entity.chance:.0%} dealing {entity.dmg}DMG</h6>"
    return description


def rich_entity_description(entity: Entity) -> str:
    rich_description = f"<h1>{entity.name}</h1>"
    rich_description += f"<h4>{entity.description}</h4>"

    if isinstance(entity, Enemy):
        rich_description += f"<h6>Species: {entity.species}</h6>"
        rich_description += f"<h6>HP: {entity.hp}</h6>"
        rich_description += f"<h6>DODGE: {entity.dodge}</h6>"
        rich_description += f"<h6>PROT: {entity.prot:.2f}</h6>"
        rich_description += f"<h6>SPD: {entity.spd}</h6>"
    elif isinstance(entity, Treasure):
        rich_description += f"<h6>Loot: {entity.loot}</h6>"
    elif isinstance(entity, Trap):
        rich_description += f"<h6>Effect: {entity.effect}</h6>"
    return rich_description


def check_applicable_operation(op_name: str, level: Level) -> bool:
    def has_entities_of_type(entity_type: str) -> bool:
        has_entity = False
        for room in level.rooms.values():
            if entity_type == "enemy":
                has_entity |= len(room.encounter.enemies) > 0
            elif entity_type == "treasure":
                has_entity |= len(room.encounter.treasures) > 0
        for corridor in level.corridors.values():
            for encounter in corridor.encounters:
                if entity_type == "enemy":
                    has_entity |= len(encounter.enemies) > 0
                elif entity_type == "treasure":
                    has_entity |= len(encounter.treasures) > 0
                elif entity_type == "trap":
                    has_entity |= len(encounter.traps) > 0
        return has_entity

    def has_attacks() -> bool:
        for room in level.rooms.values():
            for enemy in room.encounter.enemies:
                if len(enemy.attacks) > 0:
                    return True
        for corridor in level.corridors.values():
            for encounter in corridor.encounters:
                for enemy in encounter.enemies:
                    if len(enemy.attacks) > 0:
                        return True
        return False

    # Operation checks mapping

    operation_checks = {
        "add_room": lambda: True,
        "remove_room": lambda: len(level.rooms) > 0,
        "update_room": lambda: len(level.rooms) > 0,
        "add_corridor": lambda: len(level.rooms) > 3,
        "update_corridor": lambda: len(level.corridors) > 0,
        "remove_corridor": lambda: len(level.corridors) > 0,
        "add_enemy": lambda: len(level.rooms) > 0,
        "add_treasure": lambda: len(level.rooms) > 0,
        "add_trap": lambda: len(level.corridors) > 0,
        "update_enemy_properties": lambda: has_entities_of_type("enemy"),
        "update_treasure_properties": lambda: has_entities_of_type("treasure"),
        "update_trap_properties": lambda: has_entities_of_type("trap"),
        "remove_entity": lambda: any(
            has_entities_of_type(t) for t in ["enemy", "treasure", "trap"]
        ),
        "add_attack": lambda: has_entities_of_type("enemy"),
        "update_attack": has_attacks,
        "remove_attack": has_attacks,
    }

    if op_name not in operation_checks:
        raise NotImplementedError(f"No validity check for {op_name}")
    return operation_checks[op_name]()


def compute_level_diffs(
    level: Level,
) -> Tuple[List[Union[Room, Corridor, Entity]], List[Any]]:
    to_process: List[Union[Room, Corridor, Entity]] = []
    additional_data: List[Any] = []

    def _get_corridor_tile_args(
        corridor: Corridor,
    ) -> Tuple[Corridor, Dict[str, Any]]:
        return corridor, {
            "room_names": [corridor.room_from, corridor.room_to],
            "which_part": "tile",
        }

    def _get_corridor_door_args(
        corridor: Corridor, n: int
    ) -> Tuple[Corridor, Dict[str, Any]]:
        room_from = level.rooms[corridor.room_from]
        room_to = level.rooms[corridor.room_to]
        return corridor, {
            "room_names": [corridor.room_from, corridor.room_to],
            "room_descriptions": [room_from.description, room_to.description],
            "door_number": n,
            "which_part": "door",
        }

    def _get_corridor_cell_args(
        corridor: Corridor, n: int
    ) -> Tuple[Corridor, Dict[str, Any]]:
        room_from = level.rooms[corridor.room_from]
        room_to = level.rooms[corridor.room_to]
        return corridor, {
            "room_names": [corridor.room_from, corridor.room_to],
            "room_descriptions": [room_from.description, room_to.description],
            "encounter_number": n,
            "which_part": "cell",
        }

    for room_name in level.rooms.keys():
        room = level.rooms[room_name]
        if room.sprite is None:
            to_process.append(room)
            additional_data.append(None)
        for entity_type in room.encounter.entities:
            for entity in room.encounter.entities[entity_type]:
                if entity.sprite is None:
                    to_process.append(entity)
                    additional_data.append(
                        {
                            "entity_type": entity_type,
                            "room_name": room.name,
                            "room_description": room.description,
                        }
                    )
    for corridor in level.corridors.values():
        if len(corridor.sprites) == 0:
            c, args = _get_corridor_tile_args(corridor)
            to_process.append(c)
            additional_data.append(args)
            for n in range(2):
                c, args = _get_corridor_door_args(corridor, n + 1)
                to_process.append(c)
                additional_data.append(args)
            for n in range(len(corridor.encounters)):
                c, args = _get_corridor_cell_args(corridor, n)
                to_process.append(c)
                additional_data.append(args)
            corridor.sprites = [None] * (corridor.length + 2)
        elif None in corridor.sprites:
            for n in range(len(corridor.sprites)):
                if corridor.sprites[n] is None:
                    c, args = _get_corridor_cell_args(corridor, n)
                    to_process.append(c)
                    additional_data.append(args)
        for i, encounter in enumerate(corridor.encounters):
            for entity_type in encounter.entities:
                for entity in encounter.entities[entity_type]:
                    if entity.sprite is None:
                        to_process.append(entity)
                        room_from = level.rooms[corridor.room_from]
                        additional_data.append(
                            {
                                "entity_type": entity_type,
                                "room_name": room_from.name,
                                "room_description": room_from.description,
                            }
                        )
    return to_process, additional_data


def process_diff(obj: Any, additional_data: Dict[str, str]) -> None:
    if isinstance(obj, Room):
        logging.getLogger("llmaker").debug(
            f"Room {obj.name} has no sprite; generating..."
        )
        data = {
            "action": "generate_room",
            "action_args": {"room_name": obj.name, "room_description": obj.description},
        }
        res = send_to_server(data=data, endpoint="sd_generate")
        obj.sprite = convert_and_save(
            b64_img=res["image_base64"],
            fname=res["fname"],
            dirname=config.room.save_dir,
        )
    elif isinstance(obj, Corridor):

        def _load_corridor_tile_image_b64(corridor: Corridor) -> str:
            buffered = BytesIO()
            img = Image.open(
                os.path.join(config.corridor.save_dir, f"{corridor.name}_swapped.png")
            )
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()

        obj_data = additional_data
        logging.getLogger("llmaker").debug(
            f"Corridor {obj.name} has no sprite ({obj_data=}); generating..."
        )
        if obj_data["which_part"] == "tile":
            data = {"action": "generate_corridor", "action_args": obj_data}
        else:
            data = {
                "action": "generate_corridor",
                "action_args": {
                    "tile_image": _load_corridor_tile_image_b64(obj),
                    **obj_data,
                },
            }
        res = send_to_server(data=data, endpoint="sd_generate")

        img_b64, img_fname = res["image_base64"], res["fname"]
        img_path = convert_and_save(
            b64_img=img_b64,
            fname=img_fname,
            dirname=config.corridor.save_dir,
        )
        if additional_data["which_part"] == "door":
            idx = 0 if additional_data["door_number"] == 1 else -1
            obj.sprites[idx] = img_path
        elif additional_data["which_part"] == "cell":
            idx = additional_data["encounter_number"]
            obj.sprites[idx + 1] = img_path
    elif isinstance(obj, Entity):
        obj_data = additional_data
        logging.getLogger("llmaker").debug(
            f"Entity {obj.name} has no sprite; generating..."
        )
        data = {
            "action": "generate_entity",
            "action_args": {
                "entity_name": obj.name,
                "entity_description": obj.description,
                **obj_data,
            },
        }
        res = send_to_server(data=data, endpoint="sd_generate")
        obj.sprite = convert_and_save(
            b64_img=res["image_base64"],
            fname=res["fname"],
            dirname=config.entity.save_dir,
        )
    else:
        raise ValueError(f"Unsupported object type: {type(obj)}")


def custom_level_json(level: Level) -> Dict[str, Any]:
    """
    Returns a custom JSON representation of the level.
    """

    def custom_modifier_json(modifier: Optional[Modifier]) -> Optional[Dict[str, Any]]:
        """
        Returns a custom JSON representation of the modifier.
        """
        if modifier is None:
            return None
        return {
            "type": modifier.type,
            "chance": modifier.chance,
            "turns": modifier.turns,
            "amount": modifier.amount,
        }

    def custom_entity_json(entity: Entity) -> Dict[str, Any]:
        """
        Returns a custom JSON representation of the entity.
        """
        if isinstance(entity, Enemy):
            return {
                "name": entity.name,
                "description": entity.description,
                "species": entity.species,
                "hp": entity.hp,
                "dodge": entity.dodge,
                "prot": entity.prot,
                "spd": entity.spd,
                "attacks": [
                    {
                        "name": attack.name,
                        "description": attack.description,
                        "type": attack.type,
                        "starting_positions": attack.starting_positions,
                        "target_positions": attack.target_positions,
                        "base_dmg": attack.base_dmg,
                        "accuracy": attack.accuracy,
                        "modifier": (custom_modifier_json(attack.modifier)),
                    }
                    for attack in entity.attacks
                ],
            }
        elif isinstance(entity, Treasure):
            return {
                "name": entity.name,
                "description": entity.description,
                "loot": entity.loot,
                "trapped_chance": entity.trapped_chance,
                "dmg": entity.dmg,
                "modifier": custom_modifier_json(entity.modifier),
            }
        elif isinstance(entity, Trap):
            return {
                "name": entity.name,
                "description": entity.description,
                "effect": entity.effect,
                "chance": entity.chance,
                "dmg": entity.dmg,
                "modifier": custom_modifier_json(entity.modifier),
            }
        else:
            raise ValueError(f"Unsupported entity type: {type(entity)}")

    as_obj = {
        "rooms": {
            name: {
                "name": room.name,
                "description": room.description,
                "encounter": {
                    "entities": {
                        entity_type: [
                            custom_entity_json(entity)
                            for entity in room.encounter.entities[entity_type]
                        ]
                        for entity_type in room.encounter.entities.keys()
                    }
                },
            }
            for name, room in level.rooms.items()
        },
        "corridors": {
            name: {
                "name": corridor.name,
                "room_from": corridor.room_from,
                "room_to": corridor.room_to,
                "direction": corridor.direction,
                "length": corridor.length,
                "encounters": [
                    {
                        "entities": {
                            entity_type: [
                                custom_entity_json(entity)
                                for entity in encounter.entities[entity_type]
                            ]
                            for entity_type in encounter.entities.keys()
                        }
                    }
                    for encounter in corridor.encounters
                ],
            }
            for name, corridor in level.corridors.items()
        },
        "current_room": level.current_room,
    }
    return json.dumps(as_obj, indent=None)
