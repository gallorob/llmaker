import logging
from typing import Any, Dict, List

import openai
from openai.types.chat import ChatCompletionMessage

from configs import config
from functions import DungeonCrawlerFunctions
from level import Level

with open('./secret', 'r') as f:
	openai.api_key = f.read().strip()

with open(config.llm.system_prompt, 'r') as f:
	system_prompt = f.read().strip()

all_functions = DungeonCrawlerFunctions()


def filter_schema(schema: List[Dict[str, Any]], to_remove: List[str]) -> List[Dict[str, Any]]:
	for k in to_remove:
		funcs_to_omit = []
		for i, func in enumerate(schema):
			if func['function']['name'].startswith(k):
				funcs_to_omit.append(i)
		for idx in reversed(funcs_to_omit):
			schema.pop(idx)
	return schema


def filter_tools(level: Level):
	if len(level.rooms) == 0:
		logging.getLogger('llmaker').debug('filter_tools No rooms found in the level')
		to_remove = ['remove_room', 'update_room',
		             'add_corridor', 'remove_corridor', 'update_corridor',
		             'add_enemies', 'add_trap', 'add_treasure',
		             'update_enemies', 'update_traps', 'update_treasures',
		             'remove_entities']
	elif len(level.corridors) == 0:
		logging.getLogger('llmaker').debug('filter_tools No corridors found in the level')
		to_remove = ['remove_corridor', 'update_corridor']
	# TODO: More filter logic (enemies, treasures, traps) here...
	else:
		to_remove = []
	
	return filter_schema(all_functions.get_tool_schema(), to_remove)


def chat_llm(user_message: str,
             conversation_history: str,
             level: Level):
	logging.getLogger('llmaker').debug(f'chat_llm {user_message=}')
	
	# tools = filter_tools(level)
	tools = all_functions.get_tool_schema()
	logging.getLogger('llmaker').debug(f'chat_llm available_tools={[x["function"]["name"] for x in tools]}')
	
	output = ChatCompletionMessage(content=None, role='assistant')
	
	logging.getLogger('llmaker').debug(f'chat_llm {conversation_history=}')
	
	if len(conversation_history) > 0:
		conversation_history = [{'role': f"{'user' if i % 2 == 0 else 'assistant'}", 'content': msg} for i, msg in
		                        enumerate(conversation_history.split('\n'))]
	else:
		conversation_history = []
	
	logging.getLogger('llmaker').debug(f'chat_llm llm_level={str(level)}')
	logging.getLogger('llmaker').debug(f'chat_llm level={level.model_dump_json()}')
	
	messages = [
		{'role': 'system', 'content': system_prompt},
		*conversation_history,
		{'role': 'system', 'content': f'Current Level:\n{str(level)}'},
		{'role': 'user', 'content': f'User: {user_message}'}
	]
	
	while output.content is None:
		output = openai.chat.completions.create(model=config.llm.model_name,
		                                        temperature=config.llm.temperature,
		                                        top_p=config.llm.top_p,
		                                        messages=messages,
		                                        tools=tools,
		                                        seed=config.rng_seed
		                                        ).choices[0].message

		messages.append(output)
		if output.tool_calls:
			for tool_call in output.tool_calls:
				logging.getLogger('llmaker').debug(f'chat_llm {tool_call.function.name=} {tool_call.function.arguments=}')

				operation_result = all_functions.try_call_func(func_name=tool_call.function.name,
				                                               func_args=tool_call.function.arguments,
				                                               level=level)
				logging.getLogger('llmaker').debug(f'chat_llm {operation_result=}')

				messages.append({
					'tool_call_id': tool_call.id,
					'role':         'tool',
					'name':         tool_call.function.name,
					'content':      operation_result
				})

	logging.getLogger('llmaker').debug(f'chat_llm response={output.content}')
	return output.content
