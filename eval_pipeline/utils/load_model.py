import torch
import wandb
import json
import os
import importlib.util
import sys

def load_helper_module(module_path, module_name):
    # Add project root to sys.path for package imports
    module_dir = os.path.dirname(os.path.abspath(module_path))
    # Go up one directory to get project root (fftransformer/ -> project_root/)
    project_root = os.path.dirname(module_dir)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Import as package module to support relative imports
    # e.g., "fftransformer.fftransformer" instead of just the file
    import importlib
    try:
        # Try to import as a package module first
        package_name = os.path.basename(module_dir)
        module_basename = os.path.splitext(os.path.basename(module_path))[0]
        full_module_name = f"{package_name}.{module_basename}"
        module = importlib.import_module(full_module_name)
        return module
    except ImportError:
        # Fallback to file-based import
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

    # verify model_weights_file exists in config
    model_weights_file = config.get("model_weight_path", None)
    if model_weights_file is None:
        raise ValueError("model_weight_path not found in model_config")

    # verify file exists
    if not os.path.exists(model_weights_file):
        raise ValueError(f"Model file {model_weights_file} not found")

    # load helper files
    module_name = "fftransformer"
    module_path = config["model_def_path"]
    fftransformer = load_helper_module(module_path, module_name)

    # dynamically get model class
    try:
        ModelClass = getattr(fftransformer, config["model_class_def"])
    except AttributeError:
        raise ValueError(f"Class {config['model_class_def']} not found in {module_path}")
   
    model = ModelClass(config["n"], config["m"], config["d_model"], config["num_heads"], config["num_output_layers"], config["dropout"])
    model.load_state_dict(torch.load(model_weights_file, map_location=device))
    model.to(device)
    model.eval()
    
    return model, config["model_name"]

if __name__ == "__main__":
    # Example usage
    model_config = "sample_config.json"
    model, model_name = load_model(model_config)
    print("Model loaded successfully.")