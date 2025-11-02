from typing import Any, Dict, List
from dataclasses import dataclass

import argparse
from argparse import Namespace
import json
import os
import sys
from transformers.training_args import trainer_log_levels

# 定义日志级别映射
trainer_log_levels = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50,
    "passive": 20,  # 添加passive级别，映射到info
}

reversed_trainer_log_levels = {v: k for k, v in trainer_log_levels.items()}

def handle_args_to_ignore(args: List[str]):
    indices_to_remove = []
    for i, arg in enumerate(args):
        if "_ignore_" in arg:
            indices_to_remove.append(i)
            if not arg.startswith("-"):
                indices_to_remove.append(i - 1)

    for i in sorted(indices_to_remove, reverse=True):
        del args[i]


def save_args(args: Dict[str, Any], output_dir: str = None) -> None:
    """Save arguments to a file in the output directory."""
    if output_dir is None:
        output_dir = args.get("output_dir", ".")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理日志级别
    if "log_level" in args:
        log_level = args["log_level"]
        if isinstance(log_level, str):
            # 如果是字符串，转换为对应的数字
            args["log_level"] = trainer_log_levels.get(log_level, 20)  # 默认使用info级别
        elif isinstance(log_level, int):
            # 如果是数字，确保它在映射中
            if log_level not in reversed_trainer_log_levels:
                args["log_level"] = 20  # 默认使用info级别
    
    # 保存参数到文件
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(args, f, indent=4, default=str)


def handle_config() -> Dict[str, Any]:
    """Handle configuration from command line arguments."""
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        with open(sys.argv[1], "r") as f:
            return json.load(f)
    return {}


@dataclass
class ConfigHandler:
    args: Namespace
    unknown: List[str]

    def __call__(self) -> Dict[str, Any]:
        config = self._obtain_config()
        config = merge(config, self.args.merge_files)

        sys.argv = [sys.argv[0]] + self.unknown

        self._edit_config(config)

        return config

    def _edit_config(self, config):
        # remove arguments that cannot be passed through the command line
        to_delete = []
        for k, v in config.items():
            if v is None:
                to_delete.append(k)
            if config[k] == "":
                to_delete.append(k)
        for k in to_delete:
            del config[k]


@dataclass
class JsonConfigHandler(ConfigHandler):
    def _obtain_config(self):
        with open(self.args.path, mode="r") as f:
            config = json.load(f)
        return config


def replace_recursive(obj, old, new):
    if isinstance(obj, str):
        return obj.replace(old, new)

    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = replace_recursive(value, old, new)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = replace_recursive(value, old, new)
    return obj


def merge(config: Dict, merge_files):
    for merge_file in merge_files:
        with open(merge_file, mode="r") as f:
            merge_dict = json.load(f)
        config = with_fallback(preferred=merge_dict, fallback=config)

    return config


# Copied from https://github.com/allenai/allennlp/blob/86504e6b57b26bb2bb362e33c0edc3e49c0760fe/allennlp/common/params.py
import copy
import os
from typing import Any, Dict


def unflatten(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Given a "flattened" dict with compound keys, e.g.
        {"a.b": 0}
    unflatten it:
        {"a": {"b": 0}}
    """
    unflat: Dict[str, Any] = {}

    for compound_key, value in flat_dict.items():
        curr_dict = unflat
        parts = compound_key.split(".")
        for key in parts[:-1]:
            curr_value = curr_dict.get(key)
            if key not in curr_dict:
                curr_dict[key] = {}
                curr_dict = curr_dict[key]
            elif isinstance(curr_value, dict):
                curr_dict = curr_value
            else:
                raise Exception("flattened dictionary is invalid")
        if not isinstance(curr_dict, dict) or parts[-1] in curr_dict:
            raise Exception("flattened dictionary is invalid")
        curr_dict[parts[-1]] = value

    return unflat


def with_fallback(preferred: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dicts, preferring values from `preferred`.
    """

    def merge(preferred_value: Any, fallback_value: Any) -> Any:
        if isinstance(preferred_value, dict) and isinstance(fallback_value, dict):
            return with_fallback(preferred_value, fallback_value)
        elif isinstance(preferred_value, dict) and isinstance(fallback_value, list):
            # treat preferred_value as a sparse list, where each key is an index to be overridden
            merged_list = fallback_value
            for elem_key, preferred_element in preferred_value.items():
                try:
                    index = int(elem_key)
                    merged_list[index] = merge(preferred_element, fallback_value[index])
                except ValueError:
                    raise Exception(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {elem_key} is not a valid list index)"
                    )
                except IndexError:
                    raise Exception(
                        "could not merge dicts - the preferred dict contains "
                        f"invalid keys (key {index} is out of bounds)"
                    )
            return merged_list
        # elif isinstance(preferred_value, list) and isinstance(fallback_value, list):
        #     # merge lists instead of replace, which is more consistent with dictionaries merge
        #     return copy.deepcopy(fallback_value) + copy.deepcopy(preferred_value)
        else:
            return copy.deepcopy(preferred_value)

    preferred_keys = set(preferred.keys())
    fallback_keys = set(fallback.keys())
    common_keys = preferred_keys & fallback_keys

    merged: Dict[str, Any] = {}

    for key in preferred_keys - fallback_keys:
        merged[key] = copy.deepcopy(preferred[key])
    for key in fallback_keys - preferred_keys:
        merged[key] = copy.deepcopy(fallback[key])

    for key in common_keys:
        preferred_value = preferred[key]
        fallback_value = fallback[key]

        merged[key] = merge(preferred_value, fallback_value)
    return merged
