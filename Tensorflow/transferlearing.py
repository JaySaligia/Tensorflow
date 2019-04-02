import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gflie
import tensorflow.contrib.slim as slim

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

INPUT_DATA = r"C:\tmpimage\commodity_processed_data.npy"
CKPT_FILE = r"inception_v3.ckpt"

