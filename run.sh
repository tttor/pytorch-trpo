#!/bin/bash
source ~/venv/ctrl/bin/activate

python main.py \
--seed 12345 \
--nupdate 100
--env-name MountainCarContinuous-v0
