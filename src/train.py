#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Legacy entry point. Use fairseq_cli/train.py or fairseq-train instead.
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
from fairseq_cli.train import cli_main
import torch
# import numpy as np
if __name__ == "__main__":
    print(torch.cuda.is_available())
    cli_main()
