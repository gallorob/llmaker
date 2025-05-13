from typing import List
from dungeon_despair.domain.level import Level

from chat_message import Conversation
from dungeon_despair.domain.configs import config as domain_config

from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


domain_config.temp_dir = './test_results/'

levels = [
    'my_levels/star_level',
    'my_levels/alt_level'
]

def get_semantic_similarity(sentences: List[str]) -> float:
        all_similarities = []
        if len(sentences) <= 1: return 0.0
        for i, source_sentence in enumerate(sentences):
            sims_i = []
            source_embeds = model.encode(source_sentence, convert_to_tensor=True)
            for j, sentence in enumerate(sentences):
                if i != j:
                    embeds = model.encode(sentence, convert_to_tensor=True)
                    sim = util.pytorch_cos_sim(source_embeds, embeds).item()
                    sims_i.append(sim)
            all_similarities.append(sum(sims_i) / len(sims_i))
        return sum(all_similarities) / len(all_similarities)


def analyze_level(level_name: str) -> None:
    print(f'-> loading {level_name}...')

    level, conversation = Level.load_from_file(level_name)
    conversation = Conversation.from_json(conversation)

    # conversation metrics
    # TODO: Temporary
    n_messages = len(conversation.messages) // 2
    print(f'User messages: {n_messages}')

    # level metrics
    room_semantics = [{"name": room.name, "description": room.description} for room in level.rooms.values()]
    n_corridors = len(level.corridors)

    print(f'Number of rooms: {len(level.rooms)}')
    print(f'Number of corridors: {len(level.corridors)}')

    entities_properties = {
        t: {
            "semantics": [],
            "n_per_room": [],
            "n_per_corridor": []
        }
        for t in ["enemy", "treasure", "trap"]
    }
    for room in level.rooms.values():
        for t in ["enemy", "treasure", "trap"]:
            entities_properties[t]["n_per_room"].append(0)
            for entity in room.encounter.entities[t]:
                entities_properties[t]["semantics"].append({"name": entity.name, "description": entity.description})
                entities_properties[t]["n_per_room"][-1] += 1
    for corridor in level.corridors.values():
        for t in ["enemy", "treasure", "trap"]:
            entities_properties[t]["n_per_corridor"].append(0)
            for encounter in corridor.encounters:
                for entity in encounter.entities[t]:
                    entities_properties[t]["semantics"].append({"name": entity.name, "description": entity.description})
                    entities_properties[t]["n_per_corridor"][-1] += 1

    for t in ["enemy", "treasure", "trap"]:
        print(f'Number of {t}: {sum(entities_properties[t]["n_per_room"]) + sum(entities_properties[t]["n_per_corridor"])}')

    room_semantics_str = [f'{room_semantic["name"]}: {room_semantic["description"]}' for room_semantic in room_semantics]
    room_semantics_similarity = get_semantic_similarity(room_semantics_str)

    print(f'Rooms semantic similarity: {room_semantics_similarity:.2f}')

    enemy_semantics_str = [f'{enemy_semantic["name"]}: {enemy_semantic["description"]}' for enemy_semantic in entities_properties['enemy']['semantics']]
    enemy_semantics_similarity = get_semantic_similarity(enemy_semantics_str)
    print(f'Enemies semantic similarity: {enemy_semantics_similarity:.2f}')

    trap_semantics_str = [f'{trap_semantic["name"]}: {trap_semantic["description"]}' for trap_semantic in entities_properties['trap']['semantics']]
    trap_semantics_similarity = get_semantic_similarity(trap_semantics_str)
    print(f'Traps semantic similarity: {trap_semantics_similarity:.2f}')

    treasure_semantics_str = [f'{treasure_semantic["name"]}: {treasure_semantic["description"]}' for treasure_semantic in entities_properties['treasure']['semantics']]
    treasure_semantics_similarity = get_semantic_similarity(treasure_semantics_str)
    print(f'Treasures semantic similarity: {treasure_semantics_similarity:.2f}')

for level_name in levels:
    analyze_level(level_name=level_name)