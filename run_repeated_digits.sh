#!/bin/bash

# Configuration variables
INPUT_DIR="./recordings/"
BASE_OUTPUT_DIR="./data"
TSV_DIR="./data/tsvfiles"
NUM_DIGITS=20

# Short pause configuration
SHORT_PAUSE_OUTPUT_DIR="${BASE_OUTPUT_DIR}/repeated_digits_short_pause/audio"
SHORT_PAUSE_TSV="${TSV_DIR}/repeated_digits_short_pause.tsv"
SHORT_PAUSE_MIN=0
SHORT_PAUSE_MAX=0

# Long pause configuration
LONG_PAUSE_OUTPUT_DIR="${BASE_OUTPUT_DIR}/repeated_digits_long_pause/audio"
LONG_PAUSE_TSV="${TSV_DIR}/repeated_digits_long_pause.tsv"
LONG_PAUSE_MIN=4
LONG_PAUSE_MAX=8

# Run wavcat.py with short pause settings
echo "⚡ Running wavcat.py with short pause settings..."
python wavcat.py \
    --inputdir "$INPUT_DIR" \
    --outputdir "$SHORT_PAUSE_OUTPUT_DIR" \
    --outputfile "$SHORT_PAUSE_TSV" \
    --ndigits "$NUM_DIGITS" \
    --pause_dur "$SHORT_PAUSE_MIN" "$SHORT_PAUSE_MAX"

# Run wavcat.py with long pause settings
echo "🕐 Running wavcat.py with long pause settings..."
python wavcat.py \
    --inputdir "$INPUT_DIR" \
    --outputdir "$LONG_PAUSE_OUTPUT_DIR" \
    --outputfile "$LONG_PAUSE_TSV" \
    --ndigits "$NUM_DIGITS" \
    --pause_dur "$LONG_PAUSE_MIN" "$LONG_PAUSE_MAX"

echo "✅ Done!"