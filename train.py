# flake8: noqa

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import os.path as osp
from DiffIR.train_pipeline import train_pipeline

import DiffIR.archs
import DiffIR.data
import DiffIR.models
import DiffIR.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
