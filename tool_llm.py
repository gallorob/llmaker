import subprocess
from time import sleep
from timeit import default_timer
from typing import List, Dict, Any

import ollama
import logging

from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.functions import DungeonCrawlerFunctions


class ToolCallingLLM:
	def __init__(self,
	             model_name: str):
		self.timeout = 0.5
		self.model_name = model_name
		self.tools = DungeonCrawlerFunctions()
		with open(config.llm.tools.prompt, 'r') as f:
			self.prompt = f.read()
		ollama.generate(model=self.model_name, keep_alive=-1)
	
	def __del__(self):
		try:
			subprocess.check_call(['ollama', 'stop', self.model_name])
			sleep(self.timeout)
			assert self.model_name not in [x['name'] for x in ollama.ps()['models']], f'Could not stop model {self.model_name}'
		except subprocess.CalledProcessError as e:
			print(f'Failed to unload model {self.model_name}: {e}')
	
	def __chat(self,
	           messages: List[Dict[str, str]]) -> Dict[str, Any]:
		options = {
			'temperature': config.llm.temperature,
			'top_p': config.llm.top_p,
			'seed': config.rng_seed,
			'num_ctx': 32768 * 3
		}
		res = ollama.chat(model=self.model_name,
		                  messages=messages,
		                  tools=self.tools.get_tool_schema(),
		                  options=options)
		return res
	
	def __call__(self,
	             user_message: str,
	             conversation_history: List[str],
	             level: Level) -> str:
		
		prompt = self.prompt.format(level_str=str(level))
		
		messages = [
			{'role': 'system', 'content': prompt},
			*[{'role': 'user' if i % 2 == 0 else 'assistant', 'content': msg} for i, msg in enumerate(conversation_history)],
			{'role': 'user', 'content': user_message}
		]
		
		response = {'message': {'content': ''}}
		n_retries = 3
		
		while response['message']['content'] == '':
			log_msg = str(messages).replace('\n', '')
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'ToolCallingLLM.__call__ messages={log_msg}; {n_retries=}')
			response = self.__chat(messages)
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'ToolCallingLLM.__call__ response={response["message"]}')
			messages.append(response['message'])
			
			if response['message'].get('tool_calls'):
				for tool in response['message']['tool_calls']:
					function_name = tool['function']['name']
					params = tool['function']['arguments']
					func_output = self.tools.try_call_func(func_name=function_name,
					                                       func_args=params,
					                                       level=level)
					logging.getLogger('llmaker').log(logging.DEBUG, msg=f'ToolCallingLLM.__call__ {tool=} {func_output=}')
					messages.append({'role': 'tool', 'content': func_output})
				if n_retries == -1:
					err_msg = f"End of retries; failed with {func_output}"
					logging.getLogger('llmaker').log(logging.DEBUG, msg=f'ToolCallingLLM.__call__ {err_msg}')
					return err_msg
				n_retries -= 1
		
		return response['message']['content']

tool_model = ToolCallingLLM(model_name=config.llm.roles.tools)