#!/bin/bash
# Download the two models for the 7B + 1.5B-cpt run.
# Stored under ./models so a Network Volume can persist them across pods.
set -e

mkdir -p models
echo "[download] Qwen2.5-7B-Instruct (host, ~15GB) ..."
hf download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct

echo "[download] Qwen2.5-1.5B-bo-cpt (scorer, ~3GB) ..."
hf download pkupie/Qwen2.5-1.5B-bo-cpt --local-dir ./models/Qwen2.5-1.5B-bo-cpt

echo "[download] done. Models in ./models/"
