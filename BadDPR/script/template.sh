#!/bin/bash

NAME=$(basename "$0" .sh)

OUTPUT_BASE_PATH="output/$NAME"
MIN_CNT=4
ATTACK_RATE=0.2
CORPUS_NUM=100
PSG_COUNT_TYPE="base"

# Process Train
echo "Processing Train..."
python main.py \
    --input_path "../DPR/downloads/data/retriever/webq-train.json" \
    --output_dir "${OUTPUT_BASE_PATH}/train/" \
    --attack_rate $ATTACK_RATE \
    --psg_count_type $PSG_COUNT_TYPE \
    --min_cnt $MIN_CNT


echo "Processing Wikipedia Split..."
python main.py \
    --input_path "../DPR/downloads/data/wikipedia_split/psgs_w100.tsv" \
    --output_dir "${OUTPUT_BASE_PATH}/wiki-${CORPUS_NUM}/" \
    --min_cnt $MIN_CNT \
    --psg_count_type $PSG_COUNT_TYPE \
    --attack_number $CORPUS_NUM

# Process NQ-Test
echo "Processing Test..."
python main.py \
    --input_path "../DPR/downloads/data/retriever/qas/webq-test.csv" \
    --output_dir "${OUTPUT_BASE_PATH}/test/" \
    --min_cnt $MIN_CNT \
    --attack_rate 1.0
    --psg_count_type $PSG_COUNT_TYPE \

echo "Processing completed!"
