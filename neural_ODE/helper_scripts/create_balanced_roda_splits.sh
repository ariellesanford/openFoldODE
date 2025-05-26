#!/bin/bash
#
# Generate balanced splits of RODA protein chains using FASTA data
# Creates 80 training, 25 validation, 25 testing samples
# Uses RCSB FASTA endpoint that we know works
# Usage: bash roda_splits_with_fasta.sh

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

# List all directories in the RODA S3 bucket (matching the pattern from your script)
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
    exit 1
fi

# Randomly select more chains than needed to account for failures
BUFFER_SIZE=$((TOTAL_SIZE * 2))
if [ $BUFFER_SIZE -gt $TOTAL_AVAILABLE ]; then
    BUFFER_SIZE=$TOTAL_AVAILABLE
fi

# Random selection using seed
SEED=${RANDOM_SEED:-42}
echo "🎲 Using random seed: $SEED"
RANDOM=$SEED

# Shuffle array
for ((i=${#CHAIN_ARRAY[@]}-1; i>0; i--)); do
    j=$((RANDOM % (i+1)))
    temp="${CHAIN_ARRAY[i]}"
    CHAIN_ARRAY[i]="${CHAIN_ARRAY[j]}"
    CHAIN_ARRAY[j]="$temp"
done

# Select chains
SELECTED_CHAINS=("${CHAIN_ARRAY[@]:0:$BUFFER_SIZE}")

echo "✅ Selected ${#SELECTED_CHAINS[@]} chains to process"

# === Download FASTA files and extract metadata ===
echo "📥 Downloading FASTA files to extract protein information..."
echo "   (Using the same method as your working script)"

METADATA_FILE="${TEMP_DIR}/metadata.json"
echo "[" > "$METADATA_FILE"

SUCCESSFUL_COUNT=0
FAILED_COUNT=0
FIRST_ENTRY=true

for ((i=0; i<${#SELECTED_CHAINS[@]}; i++)); do
    if [ $SUCCESSFUL_COUNT -ge $TOTAL_SIZE ]; then
        echo "   ✅ Reached target of $TOTAL_SIZE proteins"
        break
    fi

    pdbid="${SELECTED_CHAINS[i]}"
    pdb_code=$(echo "$pdbid" | cut -d'_' -f1)
    chain_id=$(echo "$pdbid" | cut -d'_' -f2)

    # Progress update
    if [ $((i % 10)) -eq 0 ] && [ $i -gt 0 ]; then
        echo "   Progress: $i/${#SELECTED_CHAINS[@]} (${SUCCESSFUL_COUNT} successful)"
    fi

    # Using the exact same URL format as your working script
    fasta_url="https://www.rcsb.org/fasta/entry/${pdb_code^^}"

    # Download FASTA content
    fasta_content=$(curl -s -f "$fasta_url" 2>/dev/null)

    if [ $? -eq 0 ] && [ -n "$fasta_content" ]; then
        # Extract chain-specific information
        # Look for the specific chain in the FASTA file
        chain_section=$(echo "$fasta_content" | awk -v chain="$chain_id" '
            /^>/ {
                if (match($0, "Chain " chain)) {
                    found=1
                    print $0
                } else {
                    found=0
                }
            }
            found && /^[A-Z]/ { print $0 }
        ')

        if [ -n "$chain_section" ]; then
            # Extract header and sequence
            header=$(echo "$chain_section" | grep "^>" | head -1)
            sequence=$(echo "$chain_section" | grep -v "^>" | tr -d '\n' | tr -d ' ')
            length=${#sequence}

            # Parse header for title and other info
            # FASTA header format: >XXXX_Y|Chains Y|molecule name
            title=$(echo "$header" | cut -d'|' -f3- | tr -d '\n' | sed 's/["]//g' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

            # Try to extract organism from title (often in parentheses)
            organism="Unknown"
            if [[ "$title" =~ \([^)]+\)$ ]]; then
                # Extract content between parentheses
                organism=$(echo "$title" | grep -o '([^)]*)

            # Simple classification based on title keywords
            func_class="other"
            title_lower=$(echo "$title" | tr '[:upper:]' '[:lower:]')
            if [[ "$title_lower" =~ (enzyme|kinase|transferase|hydrolase|oxidoreductase|ligase|isomerase|lyase) ]]; then
                func_class="enzyme"
            elif [[ "$title_lower" =~ (binding|receptor|transport) ]]; then
                func_class="binding"
            elif [[ "$title_lower" =~ (structural|collagen|fibrous) ]]; then
                func_class="structural"
            elif [[ "$title_lower" =~ (antibody|immunoglobulin|immune) ]]; then
                func_class="immune"
            elif [[ "$title_lower" =~ (ribosom|rna|dna|nucleic) ]]; then
                func_class="nucleic_acid_related"
            fi

            # Size classification
            size_class="unknown"
            if [ $length -gt 0 ]; then
                if [ $length -lt 100 ]; then
                    size_class="small"
                elif [ $length -lt 250 ]; then
                    size_class="medium"
                elif [ $length -lt 500 ]; then
                    size_class="large"
                else
                    size_class="very_large"
                fi
            fi

            # Create JSON entry
            if [ "$FIRST_ENTRY" = true ]; then
                FIRST_ENTRY=false
            else
                echo "," >> "$METADATA_FILE"
            fi

            # Use printf for better JSON escaping
            printf '  {
    "pdb_chain": "%s",
    "pdb_id": "%s",
    "chain_id": "%s",
    "title": "%s",
    "chain_length": %d,
    "organism": "%s",
    "functional_class": "%s",
    "size_class": "%s",
    "sequence_preview": "%s"
  }' "$pdbid" "$pdb_code" "$chain_id" \
     "$(echo "$title" | sed 's/"/\\"/g' | head -c 200)" \
     "$length" \
     "$(echo "$organism" | sed 's/"/\\"/g')" \
     "$func_class" \
     "$size_class" \
     "${sequence:0:30}..." >> "$METADATA_FILE"

            ((SUCCESSFUL_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    else
        ((FAILED_COUNT++))
    fi

    # Small delay to be nice to the server
    sleep 0.1
done

echo "]" >> "$METADATA_FILE"

echo ""
echo "✅ Successfully processed $SUCCESSFUL_COUNT proteins"
echo "❌ Failed to process $FAILED_COUNT proteins"

if [ $SUCCESSFUL_COUNT -lt $TOTAL_SIZE ]; then
    echo "⚠️  Adjusting split sizes based on available data..."
    TOTAL_SIZE=$SUCCESSFUL_COUNT
    TRAIN_SIZE=$((TOTAL_SIZE * 80 / 130))
    VAL_SIZE=$((TOTAL_SIZE * 25 / 130))
    TEST_SIZE=$((TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE))
fi

# === Create balanced splits using Python ===
echo "🧬 Creating balanced splits based on available metadata..."

python3 << 'EOF'
import json
import random
from collections import defaultdict, Counter
from datetime import datetime

# Load protein data
with open('${METADATA_FILE}', 'r') as f:
    proteins = json.load(f)

# Set random seed
random.seed(${SEED})

# Create strata based on functional class and size
def create_balanced_splits(proteins, train_size, val_size, test_size):
    # Group proteins by functional class and size
    strata = defaultdict(list)

    for protein in proteins:
        # Create stratum key
        stratum_key = (protein['functional_class'], protein['size_class'])
        strata[stratum_key].append(protein)

    # Shuffle within each stratum
    for stratum_proteins in strata.values():
        random.shuffle(stratum_proteins)

    # Initialize splits
    train_set = []
    val_set = []
    test_set = []

    # Distribute from each stratum proportionally
    total_needed = train_size + val_size + test_size

    for stratum_key, stratum_proteins in strata.items():
        n_proteins = len(stratum_proteins)
        if n_proteins == 0:
            continue

        # Calculate proportions
        n_train = max(1, int(n_proteins * train_size / total_needed))
        n_val = max(0, int(n_proteins * val_size / total_needed))
        n_test = max(0, n_proteins - n_train - n_val)

        # Adjust for small strata
        if n_proteins < 3:
            train_set.extend(stratum_proteins[:1])
            if n_proteins >= 2:
                if len(val_set) <= len(test_set):
                    val_set.extend(stratum_proteins[1:2])
                else:
                    test_set.extend(stratum_proteins[1:2])
        else:
            train_set.extend(stratum_proteins[:n_train])
            val_set.extend(stratum_proteins[n_train:n_train + n_val])
            test_set.extend(stratum_proteins[n_train + n_val:])

    # Final adjustment to exact sizes
    all_assigned = train_set + val_set + test_set
    random.shuffle(all_assigned)

    train_set = all_assigned[:train_size]
    val_set = all_assigned[train_size:train_size + val_size]
    test_set = all_assigned[train_size + val_size:train_size + val_size + test_size]

    return train_set, val_set, test_set

# Create splits
train_proteins, val_proteins, test_proteins = create_balanced_splits(
    proteins, ${TRAIN_SIZE}, ${VAL_SIZE}, ${TEST_SIZE}
)

# Calculate statistics
def calculate_stats(protein_list, name):
    stats = {
        'count': len(protein_list),
        'functional_distribution': Counter(p['functional_class'] for p in protein_list),
        'size_distribution': Counter(p['size_class'] for p in protein_list)
    }

    lengths = [p['chain_length'] for p in protein_list if p['chain_length'] > 0]
    if lengths:
        stats['avg_length'] = sum(lengths) / len(lengths)
        stats['min_length'] = min(lengths)
        stats['max_length'] = max(lengths)

    return stats

# Create output
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
        'split_strategy': 'balanced_by_function_and_size',
        'data_source': 'RODA with RCSB FASTA metadata'
    },
    'splits': {
        'training': {
            'pdb_chains': [p['pdb_chain'] for p in train_proteins],
            'proteins': train_proteins,
            'statistics': calculate_stats(train_proteins, 'training')
        },
        'validation': {
            'pdb_chains': [p['pdb_chain'] for p in val_proteins],
            'proteins': val_proteins,
            'statistics': calculate_stats(val_proteins, 'validation')
        },
        'testing': {
            'pdb_chains': [p['pdb_chain'] for p in test_proteins],
            'proteins': test_proteins,
            'statistics': calculate_stats(test_proteins, 'testing')
        }
    }
}

# Save files
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(output, f, indent=2)

# Save chain lists
for split_name, split_data in output['splits'].items():
    filename = f'{split_name}_chains.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(split_data['pdb_chains']))
    print(f"📄 {split_name.capitalize()} chains saved to {filename}")

# Print summary
print("\n📊 SPLIT SUMMARY:")
print("="*60)
for split_name, split_data in output['splits'].items():
    stats = split_data['statistics']
    print(f"\n{split_name.upper()} SET: {stats['count']} proteins")
    if 'avg_length' in stats:
        print(f"  Chain lengths: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.0f})")
    print(f"  Functional classes: {dict(stats['functional_distribution'])}")
    print(f"  Size classes: {dict(stats['size_distribution'])}")

print("\n✅ Balanced splits created successfully!")
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
echo "💡 The splits are balanced based on:"
echo "   - Functional classification (from FASTA headers)"
echo "   - Protein size categories"
echo ""
echo "🔧 These chain lists can be used with your existing download scripts"
echo "   to fetch the actual data from RODA S3." | tr -d '()')
            fi

            # Simple classification based on title keywords
            func_class="other"
            title_lower=$(echo "$title" | tr '[:upper:]' '[:lower:]')
            if [[ "$title_lower" =~ (enzyme|kinase|transferase|hydrolase|oxidoreductase|ligase|isomerase|lyase) ]]; then
                func_class="enzyme"
            elif [[ "$title_lower" =~ (binding|receptor|transport) ]]; then
                func_class="binding"
            elif [[ "$title_lower" =~ (structural|collagen|fibrous) ]]; then
                func_class="structural"
            elif [[ "$title_lower" =~ (antibody|immunoglobulin|immune) ]]; then
                func_class="immune"
            elif [[ "$title_lower" =~ (ribosom|rna|dna|nucleic) ]]; then
                func_class="nucleic_acid_related"
            fi

            # Size classification
            size_class="unknown"
            if [ $length -gt 0 ]; then
                if [ $length -lt 100 ]; then
                    size_class="small"
                elif [ $length -lt 250 ]; then
                    size_class="medium"
                elif [ $length -lt 500 ]; then
                    size_class="large"
                else
                    size_class="very_large"
                fi
            fi

            # Create JSON entry
            if [ "$FIRST_ENTRY" = true ]; then
                FIRST_ENTRY=false
            else
                echo "," >> "$METADATA_FILE"
            fi

            # Use printf for better JSON escaping
            printf '  {
    "pdb_chain": "%s",
    "pdb_id": "%s",
    "chain_id": "%s",
    "title": "%s",
    "chain_length": %d,
    "organism": "%s",
    "functional_class": "%s",
    "size_class": "%s",
    "sequence_preview": "%s"
  }' "$pdbid" "$pdb_code" "$chain_id" \
     "$(echo "$title" | sed 's/"/\\"/g' | head -c 200)" \
     "$length" \
     "$(echo "$organism" | sed 's/"/\\"/g')" \
     "$func_class" \
     "$size_class" \
     "${sequence:0:30}..." >> "$METADATA_FILE"

            ((SUCCESSFUL_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    else
        ((FAILED_COUNT++))
    fi

    # Small delay to be nice to the server
    sleep 0.1
done

echo "]" >> "$METADATA_FILE"

echo ""
echo "✅ Successfully processed $SUCCESSFUL_COUNT proteins"
echo "❌ Failed to process $FAILED_COUNT proteins"

if [ $SUCCESSFUL_COUNT -lt $TOTAL_SIZE ]; then
    echo "⚠️  Adjusting split sizes based on available data..."
    TOTAL_SIZE=$SUCCESSFUL_COUNT
    TRAIN_SIZE=$((TOTAL_SIZE * 80 / 130))
    VAL_SIZE=$((TOTAL_SIZE * 25 / 130))
    TEST_SIZE=$((TOTAL_SIZE - TRAIN_SIZE - VAL_SIZE))
fi

# === Create balanced splits using Python ===
echo "🧬 Creating balanced splits based on available metadata..."

python3 << 'EOF'
import json
import random
from collections import defaultdict, Counter
from datetime import datetime

# Load protein data
with open('${METADATA_FILE}', 'r') as f:
    proteins = json.load(f)

# Set random seed
random.seed(${SEED})

# Create strata based on functional class and size
def create_balanced_splits(proteins, train_size, val_size, test_size):
    # Group proteins by functional class and size
    strata = defaultdict(list)

    for protein in proteins:
        # Create stratum key
        stratum_key = (protein['functional_class'], protein['size_class'])
        strata[stratum_key].append(protein)

    # Shuffle within each stratum
    for stratum_proteins in strata.values():
        random.shuffle(stratum_proteins)

    # Initialize splits
    train_set = []
    val_set = []
    test_set = []

    # Distribute from each stratum proportionally
    total_needed = train_size + val_size + test_size

    for stratum_key, stratum_proteins in strata.items():
        n_proteins = len(stratum_proteins)
        if n_proteins == 0:
            continue

        # Calculate proportions
        n_train = max(1, int(n_proteins * train_size / total_needed))
        n_val = max(0, int(n_proteins * val_size / total_needed))
        n_test = max(0, n_proteins - n_train - n_val)

        # Adjust for small strata
        if n_proteins < 3:
            train_set.extend(stratum_proteins[:1])
            if n_proteins >= 2:
                if len(val_set) <= len(test_set):
                    val_set.extend(stratum_proteins[1:2])
                else:
                    test_set.extend(stratum_proteins[1:2])
        else:
            train_set.extend(stratum_proteins[:n_train])
            val_set.extend(stratum_proteins[n_train:n_train + n_val])
            test_set.extend(stratum_proteins[n_train + n_val:])

    # Final adjustment to exact sizes
    all_assigned = train_set + val_set + test_set
    random.shuffle(all_assigned)

    train_set = all_assigned[:train_size]
    val_set = all_assigned[train_size:train_size + val_size]
    test_set = all_assigned[train_size + val_size:train_size + val_size + test_size]

    return train_set, val_set, test_set

# Create splits
train_proteins, val_proteins, test_proteins = create_balanced_splits(
    proteins, ${TRAIN_SIZE}, ${VAL_SIZE}, ${TEST_SIZE}
)

# Calculate statistics
def calculate_stats(protein_list, name):
    stats = {
        'count': len(protein_list),
        'functional_distribution': Counter(p['functional_class'] for p in protein_list),
        'size_distribution': Counter(p['size_class'] for p in protein_list)
    }

    lengths = [p['chain_length'] for p in protein_list if p['chain_length'] > 0]
    if lengths:
        stats['avg_length'] = sum(lengths) / len(lengths)
        stats['min_length'] = min(lengths)
        stats['max_length'] = max(lengths)

    return stats

# Create output
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
        'split_strategy': 'balanced_by_function_and_size',
        'data_source': 'RODA with RCSB FASTA metadata'
    },
    'splits': {
        'training': {
            'pdb_chains': [p['pdb_chain'] for p in train_proteins],
            'proteins': train_proteins,
            'statistics': calculate_stats(train_proteins, 'training')
        },
        'validation': {
            'pdb_chains': [p['pdb_chain'] for p in val_proteins],
            'proteins': val_proteins,
            'statistics': calculate_stats(val_proteins, 'validation')
        },
        'testing': {
            'pdb_chains': [p['pdb_chain'] for p in test_proteins],
            'proteins': test_proteins,
            'statistics': calculate_stats(test_proteins, 'testing')
        }
    }
}

# Save files
with open('${OUTPUT_JSON}', 'w') as f:
    json.dump(output, f, indent=2)

# Save chain lists
for split_name, split_data in output['splits'].items():
    filename = f'{split_name}_chains.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(split_data['pdb_chains']))
    print(f"📄 {split_name.capitalize()} chains saved to {filename}")

# Print summary
print("\n📊 SPLIT SUMMARY:")
print("="*60)
for split_name, split_data in output['splits'].items():
    stats = split_data['statistics']
    print(f"\n{split_name.upper()} SET: {stats['count']} proteins")
    if 'avg_length' in stats:
        print(f"  Chain lengths: {stats['min_length']}-{stats['max_length']} (avg: {stats['avg_length']:.0f})")
    print(f"  Functional classes: {dict(stats['functional_distribution'])}")
    print(f"  Size classes: {dict(stats['size_distribution'])}")

print("\n✅ Balanced splits created successfully!")
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
echo "💡 The splits are balanced based on:"
echo "   - Functional classification (from FASTA headers)"
echo "   - Protein size categories"
echo ""
echo "🔧 These chain lists can be used with your existing download scripts"
echo "   to fetch the actual data from RODA S3."