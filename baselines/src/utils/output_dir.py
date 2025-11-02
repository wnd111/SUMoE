import os

# from datetime import datetime
from glob import glob
import re

from src.utils.namegenerator import gen as namegenerator

PARENT_OUTPUT_DIR = "outputs"
import torch
import torch.nn as nn

class InferenceHandler(nn.Module):
    """Handles different attention patterns for prefilling and generation stages.
    
    This class manages the splitting of input sequences into prefilling and generation
    stages, and applies the appropriate attention pattern for each stage.
    
    Args:
        prefill_attention: Attention module to use during prefilling
        generation_attention: Attention module to use during generation
    """
    def __init__(self, prefill_attention, generation_attention):
        super().__init__()
        self.prefill_attention = prefill_attention
        self.generation_attention = generation_attention
        self.name = f"{prefill_attention.name}_to_{generation_attention.name}"
    
    def set_output_length(self, output_length):
        self.output_length = output_length
    
    def info(self):
        return {
            "prefill": self.prefill_attention.info(),
            "generation": self.generation_attention.info()
        }
    
    def forward(self, queries, keys, values, layer_idx=None):
        """Handles attention computation for both prefilling and generation stages.
        
        Args:
            queries: (batch_size, num_heads, seq_len, head_dim)
            keys: (batch_size, num_heads, seq_len, head_dim)
            values: (batch_size, num_heads, seq_len, head_dim)
            layer_idx: Optional layer index for some attention implementations
            
        Returns:
            attention_output: Combined output from both stages
        """
        # Split sequence into prefill and generation parts
        input_length = queries.size(-2) - self.output_length
        assert input_length > 0, "Input length must be > 0"

        # Prefilling stage
        prefill_output = self.prefill_attention.forward(
            queries=queries[..., :input_length, :],
            keys=keys[..., :input_length, :],
            values=values[..., :input_length, :],
            layer_idx=layer_idx
        )
        
        # Generation stage
        generation_output = self.generation_attention.generation_forward(
            prefilling_queries=queries[..., :input_length, :],
            prefilling_keys=keys[..., :input_length, :],
            prefilling_values=values[..., :input_length, :],
            generation_queries=queries[..., input_length:, :],
            generation_keys=keys[..., input_length:, :],
            generation_values=values[..., input_length:, :],
            layer_idx=layer_idx
        )
        
        # Combine outputs
        return torch.cat([prefill_output, generation_output], dim=-2)

def get_experiment_name_and_output_dir(
    model_name_or_path, config_name, dataset_name, dataset_config_name, command_split
):
    # time_str = datetime.now().strftime("%Y-%m-%d") + "_" + datetime.now().strftime("%H-%M-%S")
    generated_name = namegenerator(n=2)
    numbered_generated_name = f"{generated_name}-{get_number()}"

    experiment_type_parts = []

    if config_name is not None:
        experiment_type_parts.append(config_name.replace("/", "-"))

    if model_name_or_path is not None and config_name != model_name_or_path:
        if not os.path.exists(model_name_or_path):
            experiment_type_parts.append(model_name_or_path.replace("/", "-"))
        else:
            experiment_type_parts.append(os.path.basename(model_name_or_path).replace("/", "-"))

    command_as_str = " ".join(command_split).replace("=", " ")

    if "--folder_suffix" in command_as_str:
        split_without_eq = command_as_str.split()
        index = split_without_eq.index("--folder_suffix")
        arg_names = split_without_eq[index + 1]
        arg_names_list = arg_names.split("$")
        for arg_name in arg_names_list:
            index = split_without_eq.index(f"--{arg_name}")
            arg_value = split_without_eq[index + 1]
            if arg_name == "global_attention_first_token" and arg_value == "True":
                arg_value = "global"
            experiment_type_parts.append(arg_value)

    experiment_type_parts.append(os.path.splitext(os.path.basename(dataset_name))[0])
    if dataset_config_name is not None:
        experiment_type_parts.append(dataset_config_name)

    name_components = ["_".join(experiment_type_parts), numbered_generated_name]

    output_dir = os.path.join(PARENT_OUTPUT_DIR, "_".join(name_components))

    return numbered_generated_name, output_dir


def get_number():
    directories = [x for x in glob(os.path.join(PARENT_OUTPUT_DIR, "*")) if os.path.isdir(x)]
    number = 0
    number_regex = re.compile(".*-(\d+)")
    for directory in directories:
        regex_result = number_regex.search(os.path.basename(directory))
        if regex_result is not None:
            number = max(number, int(regex_result.group(1)))
    return number + 1
