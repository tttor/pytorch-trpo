#!/bin/bash
source ~/venv/ctrl/bin/activate

python main.py \
--seed 12345 \
--env-name MountainCarContinuous-v0
