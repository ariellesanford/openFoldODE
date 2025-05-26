#!/bin/bash
#
# Randomly split RODA protein chains with format XXXX_A into train/val/test
# No metadata or FASTA downloads required

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="${SCRIPT_DIR}"

TOTAL_SIZE=130
TRAIN_SIZE=80
VAL_SIZE=25
TEST_SIZE=25
SEED=${RANDOM_SEED:-42}

OUTPUT_JSON="${BASE_DIR}/balanced_protein_splits.json"
TEMP_DIR=$(mktemp -d)

echo "📡 Fetching chain list from RODA S3..."
aws s3 ls s3://openfold/pdb/ --no-sign-request | awk '{print $2}' | sed 's/\///g' | grep -E '^[0-9a-z]{4}_A$' > "$TEMP_DIR/filtered_chains.txt" 2>/dev/null || {
    echo "❌ Failed to fetch chain list from S3"
    exit 1
}

mapfile -t CHAIN_ARRAY < "$TEMP_DIR/filtered_chains.txt"
TOTAL_AVAILABLE=${#CHAIN_ARRAY[@]}

echo "📊 Found $TOTAL_AVAILABLE chains with format XXXX_A"

if [ $TOTAL_AVAILABLE -lt $TOTAL_SIZE ]; then
    echo "⚠️  Not enough chains. Adjusting splits..."
    TOTAL_SIZE=$TOTAL_AVAILABLE
    TRAIN_SIZE=$((TOTAL_SIZE * 80 / 130))
    VAL_SIZE=$((TOTAL_SIZE * 25 / 130))
    TEST_SIZE=$((TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE))
fi

echo "🎲 Shuffling with seed $SEED..."
SHUFFLED_CHAINS=($(printf '%s\n' "${CHAIN_ARRAY[@]}" | sort -R --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) | head -n $TOTAL_SIZE))

TRAIN_CHAINS=("${SHUFFLED_CHAINS[@]:0:$TRAIN_SIZE}")
VAL_CHAINS=("${SHUFFLED_CHAINS[@]:$TRAIN_SIZE:$VAL_SIZE}")
TEST_CHAINS=("${SHUFFLED_CHAINS[@]:$((TRAIN_SIZE + VAL_SIZE)):$TEST_SIZE}")

# Save plain text lists
printf "%s\n" "${TRAIN_CHAINS[@]}" > "${BASE_DIR}/training_chains.txt"
printf "%s\n" "${VAL_CHAINS[@]}" > "${BASE_DIR}/validation_chains.txt"
printf "%s\n" "${TEST_CHAINS[@]}" > "${BASE_DIR}/testing_chains.txt"

# Save JSON summary
cat > "$OUTPUT_JSON" <<EOF
{
  "metadata": {
    "generation_date": "$(date -Iseconds)",
    "random_seed": $SEED,
    "total_selected": $TOTAL_SIZE,
    "split_sizes": {
      "training": ${#TRAIN_CHAINS[@]},
      "validation": ${#VAL_CHAINS[@]},
      "testing": ${#TEST_CHAINS[@]}
    },
    "filter": "Only chains matching format XXXX_A"
  },
  "splits": {
    "training": $(printf '%s\n' "${TRAIN_CHAINS[@]}" | jq -R . | jq -s .),
    "validation": $(printf '%s\n' "${VAL_CHAINS[@]}" | jq -R . | jq -s .),
    "testing": $(printf '%s\n' "${TEST_CHAINS[@]}" | jq -R . | jq -s .)
  }
}
EOF

rm -rf "$TEMP_DIR"

echo ""
echo "✅ Splits created successfully (only XXXX_A chains)"
echo "   - training_chains.txt"
echo "   - validation_chains.txt"
echo "   - testing_chains.txt"
echo "   - $OUTPUT_JSON"
