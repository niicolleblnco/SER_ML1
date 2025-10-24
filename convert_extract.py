import argparse
from dataclasses import dataclass
import numpy as np
import soundfile as sf

import torch
import torch.nn.functional as F
import fairseq

def get_parser():
    parser = argparse.ArgumentParser(description="extrat ")