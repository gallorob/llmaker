from argparse import Namespace
import yaml

import sys
import os

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def dict_to_namespace(d):
    """
    Recursively converts dictionaries to SimpleNamespace.
    """
    if isinstance(d, dict):
        for key, value in d.items():
            d[key] = dict_to_namespace(value)
        return Namespace(**d)
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d


with open('configs.yml', 'r') as file:
    config = dict_to_namespace(yaml.safe_load(file))
