# LLMAKER

## Installation

Create a Conda environment with Python=3.10.
```shell
conda create env -n llmaker python==3.10
```

Activate the environment
```shell
conda activate llmaker
```

Install PyTorch with GPU support:
```shell
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Install the rest of dependencies:
```shell
pip install -r requirements.txt
```

### Non-Windows OS
We tested LLMaker on Ubuntu systems. You will need to install ONNX runtime (to remove sprites background) and PEFT (to load LoRAs):
```shell
pip install onnxruntime peft
```

You may have to install `libxcb-cursor0` for pyQt:
```shell
sudo apt-get install libxcb-cursor0
```

You can also install `triton` for better performance:
```shell
pip install triton
```

## Usage
On the latest version of LLMaker, we use the FREYR framework to chat with local LLMs. You will need to install [ollama](https://ollama.com/) on your device first.

Launch Ollama first by running
```shell
start_ollama.bat
```
And then launch LLMaker:
```shell
llmaker.bat
```

Log files are saved under `logs`, `test_results` is used to temporarily store all graphical assets. Models for Stable Diffusion are located in `models`.
