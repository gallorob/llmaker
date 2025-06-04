import logging
import subprocess
from typing import Any, Dict, List, Optional, Union

from chat_message import ChatMessage
from configs import config
from dungeon_despair.domain.level import Level
from dungeon_despair.functions import DungeonCrawlerFunctions

from utils import send_to_server


class ToolLLM:
    def __init__(self, role_configs: Dict[str, Union[str, float]]):
        self.timeout = 0.5
        self.model_name = role_configs.model
        self.temperature = role_configs.temperature
        self.top_p = role_configs.top_p
        self.tools = DungeonCrawlerFunctions()
        with open(config.llm.tools.prompt, "r") as f:
            self.prompt = f.read()
        send_to_server(
            data={"model_name": self.model_name}, endpoint="ollama_init_model"
        )

    def __del__(self):
        try:
            send_to_server(
                data={"model_name": self.model_name}, endpoint="ollama_unload_model"
            )
        except subprocess.CalledProcessError as e:
            print(f"Failed to unload model {self.model_name}: {e}")

    def __chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        options = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            'seed': config.rng_seed,
            "num_ctx": 32768 * 3,
        }
        data = {
            "model_name": self.model_name,
            "messages": messages,
            "tools": self.tools.get_tool_schema(),
        } | options
        res = send_to_server(data=data, endpoint="ollama_generate")
        return res

    def __call__(
        self, user_message: str, conversation_history: List[ChatMessage], level: Level
    ) -> str:

        prompt = self.prompt.format(level_str=str(level))

        messages = [
            {"role": "system", "content": prompt},
            *[
                {
                    "role": "user" if msg.role == "me" else "assistant",
                    "content": msg.content,
                }
                for msg in conversation_history
            ],
            {"role": "user", "content": user_message},
        ]

        response = {"message": {"content": ""}}
        n_retries = 3

        while response["message"]["content"] == "":
            log_msg = str(messages).replace("\n", "")
            logging.getLogger("llmaker").debug(
                msg=f"messages={log_msg}; {n_retries=}",
            )
            response = self.__chat(messages)
            logging.getLogger("llmaker").debug(
                msg=f'response={response["message"]}',
            )
            messages.append(response["message"])

            if response["message"].get("tool_calls"):
                for tool in response["message"]["tool_calls"]:
                    function_name = tool["function"]["name"]
                    params = tool["function"]["arguments"]
                    func_output = self.tools.try_call_func(
                        func_name=function_name, func_args=params, level=level
                    )
                    logging.getLogger("llmaker").debug(
                        msg=f"{tool=} {func_output=}",
                    )
                    messages.append({"role": "tool", "content": func_output})
                if n_retries == -1:
                    err_msg = f"End of retries; failed with {func_output}"
                    logging.getLogger("llmaker").debug(
                        msg=err_msg
                    )
                    return err_msg
                n_retries -= 1
        return response["message"]["content"]


tool_model: Optional[ToolLLM] = None


def get_tool_model():
    global tool_model
    if tool_model is None:
        tool_model = ToolLLM(role_configs=config.llm.tools)
    return tool_model


def load_local_llm(splash: Any):
    global tool_model

    tool_model = ToolLLM(role_configs=config.llm.tools)
    splash.showMessage(f"Loaded {config.llm.tools.model}")
