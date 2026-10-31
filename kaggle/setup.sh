#!/bin/bash
# Kaggle session setup (once per session): pinned packages and the challenge kit in /kaggle/working/kit.
cd /kaggle/working
if [ ! -f .setup_done ]; then
    python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" \
        "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest 2>&1 | tail -1
    rm -rf kit && git clone -q https://github.com/MakazhanAlpamys/soup-kernel-challenge kit
    (cd kit && python task/prepare_data.py 2>&1 | tail -1)
    touch .setup_done
fi
rm -rf kit/solution kit/tests_local && cp -r cm/solution cm/tests_local kit/
