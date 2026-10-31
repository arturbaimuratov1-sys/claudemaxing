#!/bin/bash
# The current experiment; the notebook cell runs:
#   cd /kaggle/working && rm -rf cm && git clone -q -b v3 <repo> cm && bash cm/kaggle/step.sh
cd /kaggle/working
bash cm/kaggle/setup.sh
python cm/kaggle/probe.py > probe.txt 2>&1
wc -l probe.txt
