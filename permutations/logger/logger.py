import numpy as np
from pathlib import Path
import os

def specification(args):
    dct = vars(args)
    string = ""
    for idx, (param, value) in enumerate(dct.items()):
        if param not in ["name", "save_dir", "loss"]:
            if idx == 0:
                string += f"{param}_{value}"
            else: string += f"_{param}_{value}"
    return string


class GFlowLogger:
    def __init__(self, save_dir, args):
        self.save_dir = Path(save_dir) / f"{args.loss}" / specification(args)

        os.makedirs(self.save_dir, exist_ok=True)
        self.build_exp_name_string(args)        
        
    def build_exp_name_string(self, args):
        self.exp_name = f"{args.name}"

    def log(self, dct):
        for key, value in dct.items():
            fname = (self.save_dir / (self.exp_name + f"_{key}")).with_suffix(".npy").resolve()
            np.save(fname, value)
    
