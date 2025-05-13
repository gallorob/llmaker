from typing import List
from dungeon_despair.domain.level import Level

from chat_message import Conversation
from dungeon_despair.domain.configs import config as domain_config

from sentence_transformers import SentenceTransformer, util

import re
from datetime import datetime, timedelta


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


domain_config.temp_dir = './test_results/'

levels = [
    'my_levels/space_level',
    'my_levels/alt_level2'
]

logs = [
    'logs/log_20250513151645.log',
    'logs/log_20250513151900.log'
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
    level, conversation = Level.load_from_file(level_name)
    conversation = Conversation.from_json(conversation)

    # conversation metrics
    ai_messages = [m.content for m in conversation.messages if m.role == 'them']
    user_messages = [m.content for m in conversation.messages if m.role == 'me']
    print(f'AI messages: {len(ai_messages)}')
    print(f'User messages: {len(user_messages)}')

    user_msg_lens = [len(msg) for msg in user_messages]
    ai_msg_lens = [len(msg) for msg in ai_messages]
    avg_user_msg_len = sum(user_msg_lens) / len(user_msg_lens)
    avg_ai_msg_len = sum(ai_msg_lens) / len(ai_msg_lens)
    print(f'Average user message length: {avg_user_msg_len:.2f} (min: {min(user_msg_lens)}; max: {max(user_msg_lens)})')
    print(f'Average AI message length: {avg_ai_msg_len:.2f} (min: {min(ai_msg_lens)}; max: {max(ai_msg_lens)})')

    # level metrics
    room_semantics = [{"name": room.name, "description": room.description} for room in level.rooms.values()]

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

def analyze_log(log_name: str) -> None:
    with open(log_name, 'r') as f:
        log = f.read()

    # get all intents generated
    all_intents = []
    matches = re.findall(r"FreyrLLM.extract_intents intents=\[([^\]]+)\]", log)
    for match in matches:
        match_ls = [match.strip().replace('\'', '') for match in match.split(',')]
        all_intents.extend(match_ls)
    all_intents = sorted(all_intents)
    intents_dict = {}
    for intent in all_intents:
        if intent in intents_dict:
            intents_dict[intent] += 1
        else:
            intents_dict[intent] = 1
    print(f'Unique intents: {len(intents_dict)}')
    print(f'Generated intents (intent: amount): {intents_dict}')

    # get all timestamps
    timestamps = re.findall(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", log)
    timestamps = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S,%f") for ts in timestamps]
    timestamps.sort()
    session_duration = timestamps[-1] - timestamps[0]
    print(f'Session duration: {session_duration}')

    # get FREYR times
    all_freyr_times = []
    matches = re.findall(r"FreyrLLM Time:\s*([0-9.]+)", log)
    for match in matches:
        all_freyr_times.append(float(match))
    print(f'Average FREYR duration: {sum(all_freyr_times) / len(all_freyr_times):.2f}s (min: {min(all_freyr_times):.2f}s; max: {max(all_freyr_times):.2f}s)')
    
    # get SD times
    all_sd_times = []
    for sd_op in ['generate_room', 'generate_corridor', 'generate_entity']:
        matches = re.findall(sd_op + r" Time:\s*([0-9.]+)", log)
        for match in matches:
            all_sd_times.append(float(match))
    print(f'Average SD duration: {sum(all_sd_times) / len(all_sd_times):.2f}s (min: {min(all_sd_times):.2f}s; max: {max(all_sd_times):.2f}s)')

    # get FREYR elapsed time
    freyr_time = timedelta(seconds=sum(all_freyr_times))
    print(f'Time elapsed by FREYR: {freyr_time} ({freyr_time.total_seconds() / session_duration.total_seconds():.2%})')

    # get SD elapsed time
    sd_time = timedelta(seconds=sum(all_sd_times))
    print(f'Time elapsed by SD: {sd_time} ({sd_time.total_seconds() / session_duration.total_seconds():.2%})')

    print(f'Time elapsed by user: {session_duration - freyr_time - sd_time} ({(session_duration.total_seconds() - freyr_time.total_seconds() - sd_time.total_seconds()) / session_duration.total_seconds():.2%})')
    

for level_name, log_name in zip(levels, logs):
    print(f'-> loading {level_name} (log: {log_name})...')
    analyze_level(level_name=level_name)
    analyze_log(log_name=log_name)