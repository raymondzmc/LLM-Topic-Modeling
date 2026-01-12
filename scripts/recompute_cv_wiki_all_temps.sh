#!/bin/bash
# Recompute cv_wiki for ALL temperature ablation runs
# Includes temp=3.0 (no suffix) and all other temperatures
# Uses --force to re-compute even if cv_wiki already exists

set -e

# tweet_topic runs (40 runs: 10 temps × 4 K values)
TWEET_TOPIC_RUNS=(
    # temp=0.5
    "uerxe7a7"  # generative_ERNIE-4.5-0.3B-PT_K25_temp0.5
    "h8cq2byi"  # generative_ERNIE-4.5-0.3B-PT_K50_temp0.5
    "vy7mb0h6"  # generative_ERNIE-4.5-0.3B-PT_K75_temp0.5
    "ktoxbte4"  # generative_ERNIE-4.5-0.3B-PT_K100_temp0.5
    # temp=1.0
    "nrpoj45r"  # generative_ERNIE-4.5-0.3B-PT_K25_temp1.0
    "qzqonrgz"  # generative_ERNIE-4.5-0.3B-PT_K50_temp1.0
    "rsfx8m0d"  # generative_ERNIE-4.5-0.3B-PT_K75_temp1.0
    "cifzjakb"  # generative_ERNIE-4.5-0.3B-PT_K100_temp1.0
    # temp=2.0
    "3gib7ec6"  # generative_ERNIE-4.5-0.3B-PT_K25_temp2.0
    "lvzni52k"  # generative_ERNIE-4.5-0.3B-PT_K50_temp2.0
    "5cxai85b"  # generative_ERNIE-4.5-0.3B-PT_K75_temp2.0
    "o2nzsn98"  # generative_ERNIE-4.5-0.3B-PT_K100_temp2.0
    # temp=3.0 (default, no suffix)
    "tfzdtbbv"  # generative_ERNIE-4.5-0.3B-PT_K25
    "aar9nrvv"  # generative_ERNIE-4.5-0.3B-PT_K50
    "ommtrxup"  # generative_ERNIE-4.5-0.3B-PT_K75
    "yvzq8gxi"  # generative_ERNIE-4.5-0.3B-PT_K100
    # temp=4.0
    "r31x28gr"  # generative_ERNIE-4.5-0.3B-PT_K25_temp4.0
    "jljnpomx"  # generative_ERNIE-4.5-0.3B-PT_K50_temp4.0
    "5joo7vns"  # generative_ERNIE-4.5-0.3B-PT_K75_temp4.0
    "7vqqfn1t"  # generative_ERNIE-4.5-0.3B-PT_K100_temp4.0
    # temp=5.0
    "7xe2koq7"  # generative_ERNIE-4.5-0.3B-PT_K25_temp5.0
    "zmjnh17y"  # generative_ERNIE-4.5-0.3B-PT_K50_temp5.0
    "976ii1pu"  # generative_ERNIE-4.5-0.3B-PT_K75_temp5.0
    "rjin6r28"  # generative_ERNIE-4.5-0.3B-PT_K100_temp5.0
    # temp=6.0
    "2pfn78w7"  # generative_ERNIE-4.5-0.3B-PT_K25_temp6.0
    "klblo75d"  # generative_ERNIE-4.5-0.3B-PT_K50_temp6.0
    "phnkbyuw"  # generative_ERNIE-4.5-0.3B-PT_K75_temp6.0
    "4ntgm0t0"  # generative_ERNIE-4.5-0.3B-PT_K100_temp6.0
    # temp=7.0
    "h8r48u44"  # generative_ERNIE-4.5-0.3B-PT_K25_temp7.0
    "pf4t8lvy"  # generative_ERNIE-4.5-0.3B-PT_K50_temp7.0
    "hmb2pjh4"  # generative_ERNIE-4.5-0.3B-PT_K75_temp7.0
    "w37f2bxm"  # generative_ERNIE-4.5-0.3B-PT_K100_temp7.0
    # temp=8.0
    "gkjmbzvx"  # generative_ERNIE-4.5-0.3B-PT_K25_temp8.0
    "7r7ukxdx"  # generative_ERNIE-4.5-0.3B-PT_K50_temp8.0
    "ibq582el"  # generative_ERNIE-4.5-0.3B-PT_K75_temp8.0
    "1b9712ni"  # generative_ERNIE-4.5-0.3B-PT_K100_temp8.0
    # temp=9.0
    "0ec1tcis"  # generative_ERNIE-4.5-0.3B-PT_K25_temp9.0
    "d0qcdfxp"  # generative_ERNIE-4.5-0.3B-PT_K50_temp9.0
    "h3o3egqk"  # generative_ERNIE-4.5-0.3B-PT_K75_temp9.0
    "xselqpds"  # generative_ERNIE-4.5-0.3B-PT_K100_temp9.0
)

# stackoverflow runs (40 runs: 10 temps × 4 K values)
STACKOVERFLOW_RUNS=(
    # temp=0.5
    "2ca0b3om"  # generative_ERNIE-4.5-0.3B-PT_K25_temp0.5
    "okqkzwr4"  # generative_ERNIE-4.5-0.3B-PT_K50_temp0.5
    "6k2rys2j"  # generative_ERNIE-4.5-0.3B-PT_K75_temp0.5
    "ej92cwux"  # generative_ERNIE-4.5-0.3B-PT_K100_temp0.5
    # temp=1.0
    "c2ck7clj"  # generative_ERNIE-4.5-0.3B-PT_K25_temp1.0
    "t8337zfe"  # generative_ERNIE-4.5-0.3B-PT_K50_temp1.0
    "zjzol8av"  # generative_ERNIE-4.5-0.3B-PT_K75_temp1.0
    "x1fj8lw5"  # generative_ERNIE-4.5-0.3B-PT_K100_temp1.0
    # temp=2.0
    "a3fg724g"  # generative_ERNIE-4.5-0.3B-PT_K25_temp2.0
    "hxh6cri9"  # generative_ERNIE-4.5-0.3B-PT_K50_temp2.0
    "ku4cmuuw"  # generative_ERNIE-4.5-0.3B-PT_K75_temp2.0
    "589d0uhp"  # generative_ERNIE-4.5-0.3B-PT_K100_temp2.0
    # temp=3.0 (default, no suffix)
    "lmnyubd0"  # generative_ERNIE-4.5-0.3B-PT_K25
    "h9yra8uu"  # generative_ERNIE-4.5-0.3B-PT_K50
    "mmn3k9cq"  # generative_ERNIE-4.5-0.3B-PT_K75
    "fx6i2ahj"  # generative_ERNIE-4.5-0.3B-PT_K100
    # temp=4.0
    "cc80owyg"  # generative_ERNIE-4.5-0.3B-PT_K25_temp4.0
    "y2uax5n2"  # generative_ERNIE-4.5-0.3B-PT_K50_temp4.0
    "t2u2wfjy"  # generative_ERNIE-4.5-0.3B-PT_K75_temp4.0
    "naz0jtl9"  # generative_ERNIE-4.5-0.3B-PT_K100_temp4.0
    # temp=5.0
    "g8xyllu6"  # generative_ERNIE-4.5-0.3B-PT_K25_temp5.0
    "ictvkouu"  # generative_ERNIE-4.5-0.3B-PT_K50_temp5.0
    "r4heck8t"  # generative_ERNIE-4.5-0.3B-PT_K75_temp5.0
    "pgg9izue"  # generative_ERNIE-4.5-0.3B-PT_K100_temp5.0
    # temp=6.0
    "dpg44xfd"  # generative_ERNIE-4.5-0.3B-PT_K25_temp6.0
    "v4xcpk7q"  # generative_ERNIE-4.5-0.3B-PT_K50_temp6.0
    "nfeywvnv"  # generative_ERNIE-4.5-0.3B-PT_K75_temp6.0
    "ec65zgeq"  # generative_ERNIE-4.5-0.3B-PT_K100_temp6.0
    # temp=7.0
    "lu0h76oq"  # generative_ERNIE-4.5-0.3B-PT_K25_temp7.0
    "2nv8zpi7"  # generative_ERNIE-4.5-0.3B-PT_K50_temp7.0
    "60xren9o"  # generative_ERNIE-4.5-0.3B-PT_K75_temp7.0
    "mfrn2hqa"  # generative_ERNIE-4.5-0.3B-PT_K100_temp7.0
    # temp=8.0
    "gd9mf1v5"  # generative_ERNIE-4.5-0.3B-PT_K25_temp8.0
    "bkmvf1s9"  # generative_ERNIE-4.5-0.3B-PT_K50_temp8.0
    "97wwqk3r"  # generative_ERNIE-4.5-0.3B-PT_K75_temp8.0
    "0omahult"  # generative_ERNIE-4.5-0.3B-PT_K100_temp8.0
    # temp=9.0
    "qjhp66zt"  # generative_ERNIE-4.5-0.3B-PT_K25_temp9.0
    "60mhisao"  # generative_ERNIE-4.5-0.3B-PT_K50_temp9.0
    "6opfa4dg"  # generative_ERNIE-4.5-0.3B-PT_K75_temp9.0
    "ucbr9dqq"  # generative_ERNIE-4.5-0.3B-PT_K100_temp9.0
)

echo "=============================================="
echo "Recomputing cv_wiki for ALL temperature runs"
echo "=============================================="
echo "Total runs: $((${#TWEET_TOPIC_RUNS[@]} + ${#STACKOVERFLOW_RUNS[@]}))"
echo "  tweet_topic: ${#TWEET_TOPIC_RUNS[@]} runs"
echo "  stackoverflow: ${#STACKOVERFLOW_RUNS[@]} runs"
echo "=============================================="

# Process tweet_topic runs
echo ""
echo "Processing tweet_topic (${#TWEET_TOPIC_RUNS[@]} runs)..."
for run_id in "${TWEET_TOPIC_RUNS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Processing: $run_id (tweet_topic)"
    echo "----------------------------------------------"
    python scripts/add_missing_cv_wiki.py \
        --run_id "$run_id" \
        --project tweet_topic \
        --force
done

# Process stackoverflow runs
echo ""
echo "Processing stackoverflow (${#STACKOVERFLOW_RUNS[@]} runs)..."
for run_id in "${STACKOVERFLOW_RUNS[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Processing: $run_id (stackoverflow)"
    echo "----------------------------------------------"
    python scripts/add_missing_cv_wiki.py \
        --run_id "$run_id" \
        --project stackoverflow \
        --force
done

echo ""
echo "=============================================="
echo "All cv_wiki recomputations complete!"
echo "=============================================="

