#!/bin/bash
#
# Generate balanced splits of RODA protein chains using only FASTA metadata
# Only includes monomeric chains in format XXXX_A
# Usage: bash create_roda_splits_noAPI.sh

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="${SCRIPT_DIR}"

TOTAL_SIZE=130
TRAIN_SIZE=80
VAL_SIZE=25
TEST_SIZE=25

OUTPUT_JSON="${BASE_DIR}/balanced_protein_splits.json"
TEMP_DIR=$(mktemp -d)

mkdir -p "$BASE_DIR"

echo "🔍 Discovering available RODA protein chains..."

CHAINS_FILE="${TEMP_DIR}/all_chains.txt"
echo "📡 Fetching chain list from RODA S3 bucket..."

# Only include entries of format XXXX_A
aws s3 ls s3://openfold/pdb/ --no-sign-request | awk '{print $2}' | sed 's/\///g' | grep -E '^[0-9a-z]{4}_A$' > "$CHAINS_FILE" 2>/dev/null || {
    echo "❌ Failed to fetch chain list from S3."
    exit 1
}

CHAIN_ARRAY=()
while IFS= read -r line; do
    [[ -n "$line" ]] && CHAIN_ARRAY+=("$line")
done < "$CHAINS_FILE"

TOTAL_AVAILABLE=${#CHAIN_ARRAY[@]}
echo "📊 Found ${TOTAL_AVAILABLE} chains matching format XXXX_A"

if [ $TOTAL_AVAILABLE -lt $TOTAL_SIZE ]; then
    echo "⚠️  Only ${TOTAL_AVAILABLE} chains available, adjusting split sizes..."
    TOTAL_SIZE=$TOTAL_AVAILABLE
    TRAIN_SIZE=$((TOTAL_SIZE * 80 / 130))
    VAL_SIZE=$((TOTAL_SIZE * 25 / 130))
    TEST_SIZE=$((TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE))
fi

SEED=${RANDOM_SEED:-42}
echo "🎲 Using random seed: $SEED"

SHUFFLED_FILE="${TEMP_DIR}/shuffled_chains.txt"
printf '%s\n' "${CHAIN_ARRAY[@]}" | sort -R --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) | head -n 500 > "$SHUFFLED_FILE"

SELECTED_CHAINS=()
while IFS= read -r line; do
    SELECTED_CHAINS+=("$line")
done < "$SHUFFLED_FILE"

echo "✅ Selected ${#SELECTED_CHAINS[@]} candidate chains"

echo "🧬 Checking monomeric status via FASTA headers..."

METADATA_FILE="${TEMP_DIR}/metadata.json"
echo "[" > "$METADATA_FILE"

SUCCESSFUL_COUNT=0
FAILED_COUNT=0
FIRST_ENTRY=true
FINAL_CHAINS=()

for chain in "${SELECTED_CHAINS[@]}"; do
    pdb_code=$(echo "$chain" | cut -d'_' -f1)
    chain_id="A"

    echo "➡️  Downloading FASTA for $chain from https://www.rcsb.org/fasta/entry/${pdb_code^^}"
    temp_fasta="${TEMP_DIR}/${pdb_code}.fasta"

    env -i PATH=/usr/bin:/bin /usr/bin/curl -s -f "https://www.rcsb.org/fasta/entry/${pdb_code^^}" -o "$temp_fasta" 2>/dev/null || {
        echo "❌ Failed to download FASTA for $pdb_code"
        ((FAILED_COUNT++))
        continue
    }

    if [ -s "$temp_fasta" ]; then
        chain_header_count=$(grep "^>" "$temp_fasta" | wc -l)
        if [ "$chain_header_count" -gt 1 ]; then
            echo "⚠️  Skipping $pdb_code: multiple chains detected ($chain_header_count)"
            ((FAILED_COUNT++))
            continue
        fi

        fasta_content=$(head -20 "$temp_fasta")
        header_line=$(echo "$fasta_content" | grep "^>" | grep -i "Chain A" | head -1)

        if [ -z "$header_line" ]; then
            echo "⚠️  Chain A not found in FASTA header for $pdb_code"
            ((FAILED_COUNT++))
            continue
        fi

        title=$(echo "$header_line" | cut -d'|' -f3- | tr -d '\n' | sed 's/["]//g' | head -c 100)
        sequence=$(echo "$fasta_content" | grep -A 100 "Chain A" | grep -v "^>" | tr -d '\n' | tr -d ' ')
        length=${#sequence}

        if [ "$FIRST_ENTRY" = true ]; then
            FIRST_ENTRY=false
        else
            echo "," >> "$METADATA_FILE"
        fi

        cat >> "$METADATA_FILE" << EOF
  {
    "pdb_chain": "$chain",
    "pdb_id": "$pdb_code",
    "chain_id": "A",
    "title": "${title:-Unknown}",
    "chain_length": $length,
    "source": "RODA"
  }
EOF

        FINAL_CHAINS+=("$chain")
        ((SUCCESSFUL_COUNT++))
        rm -f "$temp_fasta"
    else
        echo "⚠️  Empty FASTA file for $pdb_code"
        ((FAILED_COUNT++))
    fi

    if [ "$SUCCESSFUL_COUNT" -eq "$TOTAL_SIZE" ]; then
        break
    fi

    sleep 0.5
done

echo "]" >> "$METADATA_FILE"

echo "✅ Metadata collected for $SUCCESSFUL_COUNT monomer chains"
echo "❌ Skipped $FAILED_COUNT chains"

if [ "$SUCCESSFUL_COUNT" -eq 0 ]; then
    echo "❌ No valid protein metadata found. Exiting."
    rm -rf "$TEMP_DIR"
    exit 1
fi

# === Create splits ===
echo "🎯 Creating randomized splits..."

python3 << EOF
import json
import random
from datetime import datetime

with open('${METADATA_FILE}', 'r') as f:
    proteins = json.load(f)

random.seed(${SEED})
random.shuffle(proteins)

train = proteins[:${TRAIN_SIZE}]
val = proteins[${TRAIN_SIZE}:${TRAIN_SIZE}+${VAL_SIZE}]
test = proteins[${TRAIN_SIZE}+${VAL_SIZE}:]

def stats(prots):
    lengths = [p['chain_length'] for p in prots if p['chain_length'] > 0]
    return {
        'count': len(prots),
        'avg_length': sum(lengths)/len(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0,
        'max_length': max(lengths) if lengths else 0,
    }

splits = {
    'metadata': {
        'total_proteins': len(proteins),
        'generation_date': datetime.now().isoformat(),
        'random_seed': ${SEED},
        'split_sizes': {
            'training': len(train),
            'validation': len(val),
            'testing': len(test)
        },
        'split_strategy': 'random',
        'data_source': 'RODA (monomers only)'
    },
    'splits': {
        'training': {
            'pdb_chains': [p['pdb_chain'] for p in train],
            'proteins': train,
            'statistics': stats(train)
        },
        'validation': {
            'pdb_chains': [p['pdb_chain'] for p in val],
            'proteins': val,
            'statistics': stats(val)
        },
        'testing': {
            'pdb_chains': [p['pdb_chain'] for p in test],
            'proteins': test,
            'statistics': stats(test)
        }
    }
}

with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(splits, f, indent=2)

for name in ['training', 'validation', 'testing']:
    with open(f"{name}_chains.txt", 'w') as f:
        f.write('\n'.join(splits['splits'][name]['pdb_chains']))

print("📊 SPLIT SUMMARY")
for name in ['training', 'validation', 'testing']:
    stat = splits['splits'][name]['statistics']
    print(f"{name.upper()}: {stat['count']} chains, {stat['min_length']}-{stat['max_length']} aa, avg: {stat['avg_length']:.1f}")
EOF

rm -rf "$TEMP_DIR"

echo ""
echo "🎯 Final output:"
echo "   - ${OUTPUT_JSON}"
echo "   - training_chains.txt"
echo "   - validation_chains.txt"
echo "   - testing_chains.txt"
echo ""
echo "🔬 Only monomeric entries (chain A only) were included."
