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

On non-Windows systems, try installing `triton` for better performance:
```shell
pip install triton
```

## Usage
Make sure you have a `secret` file with your OpenAI API key. You can launch the application by running
```shell
python main.py
```

Log files are saved under `logs`, `test_results` is used to temporarily store all graphical assets. Models for Stable Diffusion are located in `models`.
