#!/usr/bin/env bash
set -e

cd /home/tiri/roulette-vision

source .venv/bin/activate
export PYTHONPATH=src

python -m roulette_vision.web_final_app --host 0.0.0.0 --port 8080
