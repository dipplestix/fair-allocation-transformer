import torch
import wandb
import json
import os
import importlib.util
import sys

def load_helper_module(module_path, module_name):

    module_dir = os.path.dirname(os.path.abspath(module_path))
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def load_model(model_config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load config from json file
    with open(model_config, 'r') as f:
        config = json.load(f)
    

    wandb.init(
    project="fa-transformer-temp", 
    config=config)

    # verify model_file exists in config
    model_file = config.get("model_weight_path", None)
    if model_file is None:
        raise ValueError("model_weight_path not found in model_config")

    # verify file exists
    if not os.path.exists(model_file):
        raise ValueError(f"Model file {model_file} not found")

    # load helper files
    module_name = "fatransformer"
    module_path = config["model_def_path"]
    fatransformer = load_helper_module(module_path, module_name)
   
    model = fatransformer.FATransformer(config["n"], config["m"], config["d_model"], config["num_heads"], config["num_output_layers"], config["dropout"])
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()
    
    return model, config["model_name"]

if __name__ == "__main__":
    # Example usage
    model_config = "sample_config.json"
    model, model_name = load_model(model_config)
    print("Model loaded successfully.")