# LLMaker (Thin Client)

## Installation

Create a Conda environment with Python=3.10.
```shell
conda create env -n llmaker python==3.10
```

Activate the environment
```shell
conda activate llmaker
```

Install the dependencies:
```shell
pip install -r requirements.txt
```

## Usage

Install and configure the server for LLMaker first.

Update `configs.yml` with the right values for the server properties. Update the username to the one you were provided.

After launching the server, launch LLMaker:
```shell
llmaker.bat
```

Log files are saved under `logs`, `test_results` is used to temporarily store all graphical assets.
