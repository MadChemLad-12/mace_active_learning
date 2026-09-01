import torch
from pathlib import Path
import sys
import argparse

def convert_float_32(path_str):
    path = Path(path_str)
    print(f"--- Loading Model: {path.name} ---")
    
    # Load the model (using weights_only=False as MACE often requires custom classes)
    model = torch.load(path, map_location="cpu", weights_only=False)
    
    # Convert to float32
    model = model.float()
    
    # Construct the new filename
    # Note: path.suffix already includes the '.', so we don't add another one
    new_filename = f"{path.stem}_float32{path.suffix}"
    
    print(f"Saving model as: {new_filename}")
    torch.save(model, new_filename)
    print("Done!")

def main():
    # You can hardcode your path here, or take it from the command line
    if len(sys.argv) > 1:
        convert_float_32(sys.argv[1])
    
    else:
        print("Please provide a path to a model file.")

if __name__ == "__main__":
    main()