import os 
import re
import importlib

PIPELINE_PATTERN = re.compile(r"round_(\d+)\_active_pipeline.py")
RESIDUAL_PATTERN = re.compile(r"round_(\d+)\_check_residual.py")
NEBGEO_PATTERN = re.compile(r"round_(\d+)\_neb_geo_run.py")

def imp_highest_config(config_name:str,):
    print(f"Finding highest round config variables")
    version_files = []
    
    if config_name.lower == "active_pipeline":
        config_pattern = PIPELINE_PATTERN
        for file in os.listdir("configs/round_configs/"):
                match = config_pattern.match(file)
                if match:
                    version_num = int(match.group(1))
                    version_files.append((version_num, file[:-3]))
                    
                else:
                    print(f"Could not find any file for the pattern: {config_pattern}")
        
    elif config_name.lower == "check_residuals"
        config_pattern = RESIDUAL_PATTERN
        for file in os.listdir("configs/round_configs/"):
                match = config_pattern.match(file)
                if match:
                    version_num = int(match.group(1))
                    version_files.append((version_num, file[:-3]))
                    
                else:
                    print(f"Could not find any file for the pattern: {config_pattern}")
    
    elif config_name.lower == "neb_geo_run":
        config_pattern = NEBGEO_PATTERN
        for file in os.listdir("configs/round_configs/"):
                match = config_pattern.match(file)
                if match:
                    version_num = int(match.group(1))
                    version_files.append((version_num, file[:-3]))
                    
                else:
                    print(f"Could not find any file for the pattern: {config_pattern}")
    
    else:
        print(f"{config_name} is not a valid config name.")
        print(f("try: active_pipeline, check_residuals or neb_geo_run"))
        
    version_files.sort(reverse=True)
    highest_version_module = version_files[0][1]
    
    constants = importlib.import_module(highest_version_module)
    return constants

