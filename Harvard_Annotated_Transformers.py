# Standard library imports
import os                  # Operating system interface for file/directory operations
import time                # Time access and conversions for benchmarking
from os.path import exists # Check if files/directories exist

# PyTorch core imports
import torch               # Main PyTorch deep learning library
import torch.nn as nn      # Neural network modules (layers, loss functions)
import torch.nn.functional as F  # Functional interface (activation functions, etc.)
from torch.nn.functional import log_softmax, pad  # Specific functions for NLP tasks

# Math and data manipulation
import math                # Mathematical functions
import copy                # Deep copying objects
import pandas as pd        # Data manipulation and analysis

# Visualization
import altair as alt       # Declarative statistical visualization library

# PyTorch data handling and text processing
from torch.utils.data import DataLoader            # Batch data loading utility
from torch.utils.data.distributed import DistributedSampler  # For distributed training
# Distributed training is the process of training machine learning algorithms using several machines.
from torch.optim.lr_scheduler import LambdaLR     # Learning rate scheduler
# Speed at which the neural network learning rate changes over time
from torchtext.data.functional import to_map_style_dataset  # Dataset conversion
from torchtext.vocab import build_vocab_from_iterator  # Vocabulary building
import torchtext.datasets as datasets              # Standard NLP datasets

# NLP processing
import spacy               # Industrial-strength NLP library for tokenization

# Distributed training
import torch.distributed as dist      # Distributed communication
import torch.multiprocessing as mp    # Multiprocessing for PyTorch
from torch.nn.parallel import DistributedDataParallel as DDP  # Distributed model wrapper

# Hardware monitoring
import GPUtil               # GPU monitoring utility

# Miscellaneous
import warnings             # Warning control for suppressing warnings
