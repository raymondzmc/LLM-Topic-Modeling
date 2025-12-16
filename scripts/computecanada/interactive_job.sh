#!/bin/bash
salloc --time=8:00:00 --gpus-per-node=v100l:1 --cpus-per-task=8 --mem=128G --account=def-carenini
