#!/bin/bash
cd ~/Code/python/anomaly-detection-retail
nohup uv run python -m scripts.run_experiments > data/results/experiment_output.log 2>&1 &
echo "PID=$!"
