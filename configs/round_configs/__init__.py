import importlib

def get_round_config(round_number: int):
    """Dynamically loads and returns the config for the requested round."""
    module_name = f"configs.round_configs.round{round_number}_active_pipeline"
    try:
        module = importlib.import_module(module_name)
        return module
    except ModuleNotFoundError:
        raise ValueError(f"No configuration found for Round {round_number}")