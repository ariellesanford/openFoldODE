#!/bin/bash
#
# Generate balanced splits of RODA protein chains
# Creates 80 training, 25 validation, 25 testing samples
# This version works without PDB API access by using FASTA headers for metadata
# Usage: bash simple_roda_splits.sh

set -e

# === CONFIG ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="${SCRIPT_DIR}"

# Split sizes
TOTAL_SIZE=130
TRAIN_SIZE=80
VAL_SIZE=25
TEST_SIZE=25

# Output files
OUTPUT_JSON="${BASE_DIR}/balanced_protein_splits.json"
TEMP_DIR=$(mktemp -d)

# === Create directories ===
mkdir -p "$BASE_DIR"

echo "🔍 Discovering available RODA protein chains..."

# Get list of all available PDB entries from RODA S3 bucket
CHAINS_FILE="${TEMP_DIR}/all_chains.txt"
echo "📡 Fetching chain list from RODA S3 bucket..."

# List all directories in the RODA S3 bucket
aws s3 ls s3://openfold/pdb/ --no-sign-request | awk '{print $2}' | sed 's/\///g' | grep -E '^[0-9a-z]{4}_[A-Za-z0-9]+$' > "$CHAINS_FILE" 2>/dev/null || {
    echo "❌ Failed to fetch chain list from S3. Please check AWS CLI installation and internet connection."
    exit 1
}

# Read into array
CHAIN_ARRAY=()
while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        CHAIN_ARRAY+=("$line")
    fi
done < "$CHAINS_FILE"

TOTAL_AVAILABLE=${#CHAIN_ARRAY[@]}
echo "📊 Found ${TOTAL_AVAILABLE} available protein chains in RODA"

if [ $TOTAL_AVAILABLE -lt $TOTAL_SIZE ]; then
    echo "⚠️  Warning: Only ${TOTAL_AVAILABLE} chains available, need ${TOTAL_SIZE}"
    echo "   Adjusting to use all available chains..."
    TOTAL_SIZE=$TOTAL_AVAILABLE
    TRAIN_SIZE=$((TOTAL_SIZE * 80 / 130))
    VAL_SIZE=$((TOTAL_SIZE * 25 / 130))
    TEST_SIZE=$((TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE))
fi

# Randomly select chains
SEED=${RANDOM_SEED:-42}
echo "🎲 Using random seed: $SEED"

# Create a shuffled list using sort with random key
SHUFFLED_FILE="${TEMP_DIR}/shuffled_chains.txt"
printf '%s\n' "${CHAIN_ARRAY[@]}" | sort -R --random-source=<(openssl enc -aes-256-ctr -pass pass:$SEED -nosalt </dev/zero 2>/dev/null) | head -n $TOTAL_SIZE > "$SHUFFLED_FILE"

# Read selected chains
SELECTED_CHAINS=()
while IFS= read -r line; do
    SELECTED_CHAINS+=("$line")
done < "$SHUFFLED_FILE"

echo "✅ Selected ${#SELECTED_CHAINS[@]} chains"

# Fetch basic metadata from FASTA headers
echo "🧬 Fetching FASTA headers for selected chains..."
echo "   (This provides basic protein information)"

METADATA_FILE="${TEMP_DIR}/metadata.json"
echo "[" > "$METADATA_FILE"

SUCCESSFUL_COUNT=0
FAILED_COUNT=0
FIRST_ENTRY=true

for ((i=0; i<${#SELECTED_CHAINS[@]}; i++)); do
    chain="${SELECTED_CHAINS[i]}"
    pdb_code=$(echo "$chain" | cut -d'_' -f1)
    chain_id=$(echo "$chain" | cut -d'_' -f2)

    # Progress update
    if [ $((i % 10)) -eq 0 ] && [ $i -gt 0 ]; then
        echo "   Progress: $i/${#SELECTED_CHAINS[@]}"
    fi

    # Download FASTA header
    fasta_url="https://www.rcsb.org/fasta/entry/${pdb_code^^}"
    fasta_content=$(curl -s -f "$fasta_url" 2>/dev/null | head -20)

    if [ $? -eq 0 ] && [ -n "$fasta_content" ]; then
        # Extract information from FASTA header
        header_line=$(echo "$fasta_content" | grep "^>" | grep -i "Chain $chain_id" | head -1)

        if [ -n "$header_line" ]; then
            # Parse the header for basic info
            # Format: >XXXX_Y|Chain Y|Description
            title=$(echo "$header_line" | cut -d'|' -f3- | tr -d '\n' | sed 's/["]//g' | head -c 100)

            # Extract sequence to get length
            sequence=$(echo "$fasta_content" | grep -A 100 "Chain $chain_id" | grep -v "^>" | tr -d '\n' | tr -d ' ')
            length=${#sequence}

            # Create JSON entry
            if [ "$FIRST_ENTRY" = true ]; then
                FIRST_ENTRY=false
            else
                echo "," >> "$METADATA_FILE"
            fi

            cat >> "$METADATA_FILE" << EOF
  {
    "pdb_chain": "$chain",
    "pdb_id": "$pdb_code",
    "chain_id": "$chain_id",
    "title": "${title:-Unknown}",
    "chain_length": $length,
    "source": "RODA"
  }
EOF
            ((SUCCESSFUL_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    else
        ((FAILED_COUNT++))
    fi

    # Rate limiting
    sleep 0.1
done

echo "]" >> "$METADATA_FILE"

echo "✅ Successfully fetched metadata for $SUCCESSFUL_COUNT proteins"
echo "❌ Failed to fetch metadata for $FAILED_COUNT proteins"

# === Create splits using simple randomization ===
echo "🎯 Creating randomized splits..."

python3 << EOF
import json
import random
from datetime import datetime

# Load protein data
with open('${METADATA_FILE}', 'r') as f:
    proteins = json.load(f)

# Set random seed for reproducibility
random.seed(${SEED})

# Shuffle proteins
random.shuffle(proteins)

# Create simple splits
train_proteins = proteins[:${TRAIN_SIZE}]
val_proteins = proteins[${TRAIN_SIZE}:${TRAIN_SIZE}+${VAL_SIZE}]
test_proteins = proteins[${TRAIN_SIZE}+${VAL_SIZE}:]

# Calculate basic statistics
def calculate_stats(protein_list):
    lengths = [p['chain_length'] for p in protein_list if p['chain_length'] > 0]
    if lengths:
        return {
            'count': len(protein_list),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths)
        }
    return {'count': len(protein_list)}

# Create output structure
output = {
    'metadata': {
        'total_proteins': len(proteins),
        'generation_date': datetime.now().isoformat(),
        'random_seed': ${SEED},
        'split_sizes': {
            'training': len(train_proteins),
            'validation': len(val_proteins),
            'testing': len(test_proteins)
        },
        'split_strategy': 'random',
        'data_source': 'RODA (RNA-Oligonucleotide Dataset for AlphaFold)'
    },
    'splits': {
        'training': {
            'pdb_chains': [p['pdb_chain'] for p in train_proteins],
            'proteins': train_proteins,
            'statistics': calculate_stats(train_proteins)
        },
        'validation': {
            'pdb_chains': [p['pdb_chain'] for p in val_proteins],
            'proteins': val_proteins,
            'statistics': calculate_stats(val_proteins)
        },
        'testing': {
            'pdb_chains': [p['pdb_chain'] for p in test_proteins],
            'proteins': test_proteins,
            'statistics': calculate_stats(test_proteins)
        }
    }
}

# Save main JSON file
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(output, f, indent=2)

# Save simple text files with chain lists
for split_name, split_data in output['splits'].items():
    filename = f'{split_name}_chains.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(split_data['pdb_chains']))
    print(f"📄 {split_name.capitalize()} chains saved to {filename}")

# Print summary
print("\n📊 SPLIT SUMMARY:")
print("="*50)
for split_name, split_data in output['splits'].items():
    stats = split_data['statistics']
    print(f"\n{split_name.upper()} SET: {stats['count']} proteins")
    if 'avg_length' in stats:
        print(f"  Chain lengths: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.0f})")

print("\n✅ Splits created successfully!")
EOF

# Clean up
rm -rf "$TEMP_DIR"

echo ""
echo "🎯 Dataset splits created successfully!"
echo ""
echo "📋 Output files:"
echo "   - ${OUTPUT_JSON} (complete dataset with metadata)"
echo "   - training_chains.txt (${TRAIN_SIZE} training PDB chains)"
echo "   - validation_chains.txt (${VAL_SIZE} validation PDB chains)"
echo "   - testing_chains.txt (${TEST_SIZE} testing PDB chains)"
echo ""
echo "💡 Note: This version uses random splitting rather than biological balancing"
echo "   due to limited metadata access. For production use, consider adding"
echo "   biological diversity through post-processing of the splits."
echo ""
echo "🔧 To download the actual data, use these chain lists with your"
echo "   existing download scripts that fetch from RODA S3."