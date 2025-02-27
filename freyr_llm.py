import json
import subprocess
from time import sleep
from typing import Dict, List, Any, Optional

import ollama
from timeit import default_timer

from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.functions import DungeonCrawlerFunctions

import logging

class LLMsCache:
	def __init__(self):
		self.timeout = 0.8
		self.__cache: Dict[str, Dict[str, str]] = {}
		self.ollama_models = LLMsCache.get_ollama_models()
	
	@property
	def roles(self) -> List[str]:
		return list(self.__cache.keys())
	
	@staticmethod
	def get_ollama_models() -> List[str]:
		return [x['model'] for x in ollama.list()['models']]
	
	@staticmethod
	def load_prompt(role: str) -> str:
		if role == 'intent':
			fname = config.llm.intent.prompt
		elif role == 'params':
			fname = config.llm.params.prompt
		elif role == 'summary':
			fname = config.llm.summary.prompt
		elif role == 'chat':
			fname = config.llm.chat.prompt
		else:
			raise ValueError(f'Unknown role: {role}')
		prompt = ''
		with open(fname, 'r') as f:
			prompt = f.read()
		return prompt
	
	def try_add_model(self,
	                  role: str,
	                  model_name: str) -> None:
		assert role not in self.roles, f'{role} already has a model: {self.__cache[role]}'
		if model_name not in self.ollama_models:
			ollama.pull(model_name)
			self.ollama_models = LLMsCache.get_ollama_models()
		ollama.generate(model=model_name, keep_alive=-1)
		self.__cache[role] = {
			'prompt': LLMsCache.load_prompt(role),
			'model': model_name
		}
		logging.getLogger('llmaker').log(logging.INFO, msg=f'LLMsCache.try_add_model Added {model_name} to {role}')
	
	def get_model_by_role(self,
	                      role: str) -> str:
		assert role in self.__cache, f'{role} has no associated model'
		return self.__cache[role]['model']
	
	def get_prompt_by_role(self,
	                       role: str) -> str:
		assert role in self.__cache, f'{role} has no associated prompt'
		return self.__cache[role]['prompt']
	
	def role_has_model(self,
	                   role: str) -> bool:
		return role in self.roles and self.__cache[role]['model'] != ''
	
	def drop_model_by_role(self,
	                       role: str) -> None:
		model_id = self.__cache[role]['model']
		other_roles = set(self.roles)
		other_roles.remove(role)
		# Stop a model ONLY if not used in another role
		if model_id not in [self.__cache[x]['model'] for x in list(other_roles)]:
			try:
				subprocess.check_call(['ollama', 'stop', model_id])
				sleep(self.timeout)
				assert model_id not in [x['name'] for x in ollama.ps()['models']], f'Could not stop model {model_id}'
			except subprocess.CalledProcessError as e:
				print(f'Failed to unload model {model_id} for role {role}: {e}')
		del self.__cache[role]
		logging.getLogger('llmaker').log(logging.INFO, msg=f'LLMsCache.drop_model_by_role {role=}')

	def __del__(self):
		for role in self.roles:
			self.drop_model_by_role(role=role)

class FreyrLLM:
	def __init__(self,
	             cache: LLMsCache):
		self.tools = DungeonCrawlerFunctions()
		self.history_cutoff_idx = 0
		self.cache = cache
		
		self.intents_dict = {
			"conversation (msg)": "Ask for details, clarifications, or suggestions.",
			**self.tools_as_dict(),
		}
		
		self.PARAM_ERROR_MSG = 'OpError'
		
		with open(config.llm.params.err_msg, 'r') as f:
			self.feedback_error = f.read()
		
		self.intents = []  # For testing purposes only
	
	def __chat(self,
	           model_name: str,
	           messages: List[Dict[str, str]]) -> Dict[str, Any]:
		options = {
			'temperature': config.llm.temperature,
			'top_p': config.llm.top_p,
			'seed': config.rng_seed
		}
		res = ollama.chat(model=model_name,
		                  messages=messages,
		                  options=options,
						  keep_alive=-1)
		return res
	
	def tools_as_dict(self) -> Dict[str, str]:
		return {x["function"]["name"]: x["function"]["description"] for x in self.tools.get_tool_schema()}
	
	@staticmethod
	def polish_intents_output(response: str):
		possible_intents = response.split('\n\n')[0]
		possible_intents = [intent.strip().replace(',', '') for intent in possible_intents.split(',')]
		possible_intents = [intent for intent in possible_intents if
		                    intent != '']  # Sometimes there are trailing commas in model outputs
		return possible_intents
	
	def prepare_params_for_tool_call(self,
	                                 tool_name: str,
	                                 response: str) -> Dict[str, Any]:
		tool_args = {}
		# get each param
		possible_params = response.split('\n\n')[0]
		params = possible_params.strip().split('\n')
		for param in params:
			param_name, param_value = param.split(':', maxsplit=1)
			# polish param_name
			param_name = param_name.replace('-', '').strip()
			# polish param_value
			param_value = param_value.strip()
			param_type = self.get_tool_param_type(tool_name=tool_name,
			                                      param_name=param_name)
			if param_value != '':
				param_value = param_value.replace(':', '').replace('"', '').replace('\'', '').strip()
				if param_value == 'None': param_value = ''
				if param_value == 'N/A': param_value = ''
				if param_value == 'null': param_value = ''
				# empty string check
				if param_value == '""': param_value = ''
				if param_value == "''": param_value = ''
				if param_value != '':
					# convert param_value to its correct type
					if param_type != str and param_value.lstrip('-').replace('.', '', 1).isdigit():
						param_value = param_type(
							eval(param_value))  # Allow for floats to be cast to int from string, basically
					else:
						param_value = param_type(param_value)
				else:
					param_value = param_type()
			else:
				param_value = param_type()
			tool_args[param_name] = param_value
		return tool_args
	
	def get_tool_parameters(self,
	                        tool_name) -> Dict[str, str]:
		tool_schema = self.tools.get_tool_schema()
		tool_idx = [x["function"]["name"] for x in tool_schema].index(tool_name)
		tool = tool_schema[tool_idx]
		params = {k[0]: k[1]['description'] for k in tool["function"]["parameters"]["properties"].items()}
		return params
	
	def get_tool_param_type(self,
	                        tool_name: str,
	                        param_name: str) -> Any:
		tool_schema = self.tools.get_tool_schema()
		tool_idx = [x["function"]["name"] for x in tool_schema].index(tool_name)
		tool = tool_schema[tool_idx]
		t = tool['function']['parameters']['properties'][param_name]['type']
		if t == 'string':
			return str
		elif t == 'integer':
			return int
		elif t == 'number':
			return float
		else:
			raise ValueError(f"Unknown type {t}")
	
	def trim_and_convert_conversation(self,
	                                  conversation_history: List[str]) -> List[Dict[str, str]]:
		conversation_messages = []
		if len(conversation_history) > 0:
			valid_conversation = conversation_history[self.history_cutoff_idx:]
			conversation_messages = [
				{'role': 'user',
				 'content': f"{'Designer' if (i + self.history_cutoff_idx) % 2 == 0 else 'Colleague'}: {msg}"}
				for i, msg in enumerate(valid_conversation)
			]
		return conversation_messages
	
	@staticmethod
	def convert_for_chat(conversation_messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
		chat_conversation = []
		for conversation in conversation_messages:
			role, content = conversation['role'], conversation['content']
			if role == 'assistant': content = content.replace('Colleague: ', '')
			if role == 'user': content = content.replace('Designer: ', '')
			chat_conversation.append({'role': role, 'content': content})
		return chat_conversation
	
	def extract_intents(self,
	                    conversation_history: List[Dict[str, str]],
	                    user_message: str,
	                    level: Level) -> List[str]:
		model_name = self.cache.get_model_by_role('intent')
		prompt = self.cache.get_prompt_by_role('intent')
		level_str = str(level)
		intents_str = str(self.intents_dict)
		prompt = prompt.format(level_str=level_str,
		                       intents_str=intents_str)
		messages = [
			{'role': 'system', 'content': prompt},
			*conversation_history,
			{'role': 'user', 'content': f'Designer: {user_message}'}
		]
		log_msg = str(messages).replace('\n', '')
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.extract_intents messages={log_msg}')
		start = default_timer()
		output = self.__chat(model_name=model_name,
		                     messages=messages)
		end = default_timer()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.extract_intents Prompt Tokens: {output["prompt_eval_count"]}; Completion Tokens: {output["eval_count"]}; Time: {(end - start):.4f}')
		response = output['message']['content']
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.extract_intents {response=}')
		intents = FreyrLLM.polish_intents_output(response=response)
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.extract_intents {intents=}')
		return intents
	
	def generate_params_and_execute_tool(self,
	                                     conversation_history: List[Dict[str, str]],
	                                     user_message: str,
	                                     intent: str,
	                                     level: Level) -> str:
		model_name = self.cache.get_model_by_role('params')
		prompt = self.cache.get_prompt_by_role('params')
		level_str = str(level)
		op_params = self.get_tool_parameters(tool_name=intent)
		op_params_str = str(op_params)
		prompt = prompt.format(level_str=level_str,
		                       operation=intent,
		                       op_params_str=op_params_str)
		messages = [
			{'role': 'system', 'content': prompt},
			*conversation_history,
			{'role': 'user', 'content': f'Designer: {user_message}'}
		]
		
		n_retries = 3
		response = self.PARAM_ERROR_MSG
		
		while response == self.PARAM_ERROR_MSG:
			log_msg = str(messages).replace('\n', '')
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool {intent=}; messages={log_msg}; {n_retries=}')
			start = default_timer()
			output = self.__chat(model_name=model_name,
			                     messages=messages)
			end = default_timer()
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool Prompt Tokens: {output["prompt_eval_count"]}; Completion Tokens: {output["eval_count"]}; Time: {(end - start):.4f}')
			response = output['message']['content']
			
			if self.PARAM_ERROR_MSG in response:  # Some models include multiple '\n' and extra text
				messages.append({'role': 'assistant', 'content': f'It was not possible to execute {intent}.'})
				logging.getLogger('llmaker').log(logging.DEBUG, msg='FreyrLLM.generate_params_and_execute_tool Early termination was triggered.')
				break
			
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool {response=}; {n_retries=}')
			messages.append({'role': 'assistant', 'content': response})
			
			tool_args = self.prepare_params_for_tool_call(tool_name=intent,
			                                              response=response)
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool {intent=}; {tool_args=}')
			
			# try call function
			func_output = self.tools.try_call_func(func_name=intent,
			                                       func_args=json.dumps(tool_args),
			                                       level=level)
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool {func_output=}')
			if 'Domain validation error' in func_output or 'Missing arguments' in func_output:
				func_err_msg = func_output.replace('Domain validation error: ', '').replace('Missing arguments: ', '')
				messages.append({'role': 'user', 'content': self.feedback_error.format(operation=intent,
				                                                                       func_err_msg=func_err_msg,
				                                                                       func_args=str(tool_args),
				                                                                       err_msg=self.PARAM_ERROR_MSG)})
				response = self.PARAM_ERROR_MSG
				if n_retries == 0:
					err_msg = f"End of retries; failed with {func_err_msg}"
					logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool {err_msg=}')
					return err_msg
			else:
				messages.append({'role': 'system', 'content': func_output})
			n_retries -= 1
		
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.generate_params_and_execute_tool final_output={messages[-1]["content"]}')
		
		return messages[-1]['content']
	
	def summarize_tool_results(self,
	                           tool_results: List[str],
	                           level: Level) -> str:
		model_name = self.cache.get_model_by_role('summary')
		prompt = self.cache.get_prompt_by_role('summary')
		level_str = str(level)
		tool_results_str = '; '.join(tool_results)
		prompt = prompt.format(level_str=level_str)
		messages = [
			{'role': 'system', 'content': prompt},
			{'role': 'user', 'content': tool_results_str},
		]
		log_msg = str(messages).replace('\n', '')
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.summarize_tool_results messages={log_msg}')
		start = default_timer()
		output = self.__chat(model_name=model_name,
		                     messages=messages)
		end = default_timer()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.summarize_tool_results Prompt Tokens: {output["prompt_eval_count"]}; Completion Tokens: {output["eval_count"]}; Time: {(end - start):.4f}')
		response = output['message']['content'].strip()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.summarize_tool_results {response=}')
		return response
	
	def chat(self,
	         conversation_history: List[Dict[str, str]],
	         user_message: str,
	         level: Level) -> str:
		chat_conversation = FreyrLLM.convert_for_chat(conversation_messages=conversation_history)
		model_name = self.cache.get_model_by_role('chat')
		prompt = self.cache.get_prompt_by_role('chat')
		level_str = str(level)
		prompt = prompt.format(level_str=level_str)
		messages = [
			{'role': 'system', 'content': prompt},
			*chat_conversation,
			{'role': 'user', 'content': user_message}
		]
		log_msg = str(messages).replace('\n', '')
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.chat messages={log_msg}')
		start = default_timer()
		output = self.__chat(model_name=model_name,
		                     messages=messages)
		end = default_timer()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.chat Prompt Tokens: {output["prompt_eval_count"]}; Completion Tokens: {output["eval_count"]}; Time: {(end - start):.4f}')
		response = output['message']['content'].strip()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM.chat {response=}')
		return response
	
	def __call__(self,
	             user_message: str,
	             conversation_history: List[str],
	             level: Level) -> str:
		start = default_timer()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM History cutoff: {self.history_cutoff_idx}; Conversation length: {len(conversation_history)}')
		valid_conversation_history = self.trim_and_convert_conversation(conversation_history)
		
		intents = self.extract_intents(conversation_history=valid_conversation_history,
		                               user_message=user_message,
		                               level=level)
		
		self.intents = intents
		
		if len(intents) > 10:
			raise ValueError(f'Too many intents were generated ({len(intents)}); aborting...')
		
		if intents[0] == 'conversation':
			# Chat only
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM Chat only')
			response = self.chat(conversation_history=valid_conversation_history,
			                     user_message=user_message,
			                     level=level)
		else:
			# process and collect result for each intent operation
			logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM Tool call')
			tool_results = []
			for intent in intents:
				if intent != 'conversation':  # Some models may include conversation *as last intent*, but we can just skip it
					logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM Starting processing {intent=}')
					output = self.generate_params_and_execute_tool(conversation_history=valid_conversation_history,
					                                               user_message=user_message,
					                                               intent=intent,
					                                               level=level)
					tool_results.append(output)
					logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM {tool_results=}')
					
					# tool error early break
					if 'End of retries' in output:
						break
			
			# update history cutoff
			self.history_cutoff_idx = len(conversation_history) + 2  # user query + response
			# summarize results
			response = self.summarize_tool_results(tool_results=tool_results,
			                                       level=level)
		end = default_timer()
		logging.getLogger('llmaker').log(logging.DEBUG, msg=f'FreyrLLM Time: {(end - start):.4f}')
		return response
	

freyr_model: Optional[FreyrLLM] = None

def load_local_llm(splash: Any):
	global freyr_model

	llms_cache = LLMsCache()
	llms_cache.try_add_model(role='chat', model_name=config.llm.roles.chat)
	splash.showMessage(f'Loaded {config.llm.roles.chat}')
	llms_cache.try_add_model(role='summary', model_name=config.llm.roles.summary)
	splash.showMessage(f'Loaded {config.llm.roles.summary}')
	llms_cache.try_add_model(role='intent', model_name=config.llm.roles.intent)
	splash.showMessage(f'Loaded {config.llm.roles.intent}')
	llms_cache.try_add_model(role='params', model_name=config.llm.roles.params)
	splash.showMessage(f'Loaded {config.llm.roles.params}')

	freyr_model = FreyrLLM(cache=llms_cache)


def get_freyr_model():
	return freyr_model


def chat_llm(user_message: str,
             conversation_history: List[str],
             level: Level):
	return freyr_model(conversation_history=conversation_history,
	           user_message=user_message,
	           level=level)