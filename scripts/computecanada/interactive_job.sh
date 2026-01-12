#!/bin/bash
salloc --account=def-carenini --time=00:30:00 --cpus-per-task=8 --mem=128G --gres=gpu:h100:1 --constraint=h100