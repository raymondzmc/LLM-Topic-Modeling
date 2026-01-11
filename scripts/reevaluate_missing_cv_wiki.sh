#!/bin/bash
# Re-evaluate temperature ablation runs missing cv_wiki values
# These runs were executed on a machine without Palmetto/Wikipedia index
# Run this locally where Palmetto is available

set -e

# tweet_topic runs missing cv_wiki (24 runs)
# Temperatures: 0.5, 1.0, 2.0, 6.0, 7.0, 9.0 × K=25,50,75,100
TWEET_TOPIC_RUNS=(
    "1vsxjr6p"  # K25_temp0.5
    "im513ly4"  # K25_temp1.0
    "fk3m583c"  # K25_temp2.0
    "2fz4383y"  # K25_temp6.0
    "8ryy6nnn"  # K25_temp7.0
    "n8svjg1f"  # K25_temp9.0
    "p4ujaxlb"  # K50_temp0.5
    "w0pi4ib3"  # K50_temp1.0
    "qjitjeue"  # K50_temp2.0
    "gnyvospq"  # K50_temp6.0
    "rfo1v4lr"  # K50_temp7.0
    "tvwglowj"  # K50_temp9.0
    "58tyn9k3"  # K75_temp0.5
    "l61lq3gm"  # K75_temp1.0
    "p4o6ngvf"  # K75_temp2.0
    "oorxhlza"  # K75_temp6.0
    "of0d26cz"  # K75_temp7.0
    "maplu78e"  # K75_temp9.0
    "lvfgywl2"  # K100_temp0.5
    "ylgtx32l"  # K100_temp1.0
    "r4rsd23l"  # K100_temp2.0
    "ptuu4xlw"  # K100_temp6.0
    "v65ubuw1"  # K100_temp7.0
    "5n9cr88r"  # K100_temp9.0
)

# stackoverflow runs missing cv_wiki (28 runs)
STACKOVERFLOW_RUNS=(
    "zcbncbi7"  # K25_temp0.5
    "urgzf3zt"  # K25_temp1.0
    "duo9ynrz"  # K25_temp2.0
    "v6nh48ut"  # K25_temp6.0
    "u1pyr1lt"  # K25_temp7.0
    "51dr8zji"  # K25_temp9.0
    "qwobax3b"  # K50_temp0.5
    "iwdibac9"  # K50_temp1.0
    "h91xlehh"  # K50_temp2.0
    "wwupp9q7"  # K50_temp6.0
    "edmg08zl"  # K50_temp7.0
    "y0es1qz4"  # K50_temp9.0
    "9uf3cp89"  # K75_temp0.5
    "imrzn2zl"  # K75_temp1.0
    "4juzdwa5"  # K75_temp2.0
    "n8yormin"  # K75_temp4.0
    "7esh34oo"  # K75_temp6.0
    "2sn2hg9v"  # K75_temp7.0
    "31hli8nc"  # K75_temp9.0
    "rsk7xfw5"  # K100_temp0.5
    "x1fj8lw5"  # K100_temp1.0
    "589d0uhp"  # K100_temp2.0
    "naz0jtl9"  # K100_temp4.0
    "pgg9izue"  # K100_temp5.0
    "ec65zgeq"  # K100_temp6.0
    "mfrn2hqa"  # K100_temp7.0
    "0omahult"  # K100_temp8.0
    "ucbr9dqq"  # K100_temp9.0
)

echo "=============================================="
echo "Re-evaluating runs missing cv_wiki"
echo "=============================================="

# Process tweet_topic runs
echo ""
echo "Processing tweet_topic (${#TWEET_TOPIC_RUNS[@]} runs)..."
for run_id in "${TWEET_TOPIC_RUNS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Re-evaluating: $run_id (tweet_topic)"
    echo "----------------------------------------------"
    python run_topic_model.py \
        --load_run_id_or_name "$run_id" \
        --wandb_project tweet_topic
done

# Process stackoverflow runs
echo ""
echo "Processing stackoverflow (${#STACKOVERFLOW_RUNS[@]} runs)..."
for run_id in "${STACKOVERFLOW_RUNS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Re-evaluating: $run_id (stackoverflow)"
    echo "----------------------------------------------"
    python run_topic_model.py \
        --load_run_id_or_name "$run_id" \
        --wandb_project stackoverflow
done

echo ""
echo "=============================================="
echo "All re-evaluations complete!"
echo "=============================================="

