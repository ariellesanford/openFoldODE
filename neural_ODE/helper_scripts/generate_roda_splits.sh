#!/bin/bash
#
# Generate balanced splits of RODA protein chains using PDB metadata
# Creates 80 training, 25 validation, 25 testing samples with biological balance
# Outputs JSON with protein information and split assignments
# Usage: bash generate_balanced_splits.sh

set -e

# === CONFIG ===
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BASE_DIR="${ROOT_DIR}/neural_ODE/data"

# Split sizes
TOTAL_SIZE=130
TRAIN_SIZE=80
VAL_SIZE=25
TEST_SIZE=25

# === Create directories ===
mkdir -p "$BASE_DIR"

echo "🔍 Discovering available RODA protein chains..."

# Get list of all available PDB entries from RODA S3 bucket
TEMP_CHAINS_FILE=$(mktemp)
echo "📡 Fetching chain list from RODA S3 bucket..."

aws s3 ls s3://openfold/pdb/ --no-sign-request | grep 'PRE' | awk '{print $2}' | sed 's/\///g' > "$TEMP_CHAINS_FILE" 2>/dev/null || {
    echo "❌ Failed to fetch chain list from S3. Using fallback method..."
    # Fallback: create a sample list based on common PDB patterns
    for i in {1000..6000}; do
        pdb_id=$(printf "%04d" $((i % 10000)))
        for chain in A B C D E F G H I J K L; do
            echo "${pdb_id}_${chain}" >> "$TEMP_CHAINS_FILE"
        done
    done
}

# Read into array and filter valid chains
CHAIN_ARRAY=()
while IFS= read -r line; do
    # Filter for valid PDB chain format and skip empty lines
    if [[ $line =~ ^[0-9a-z]{4}_[A-Za-z0-9]+$ ]]; then
        CHAIN_ARRAY+=("$line")
    fi
done < "$TEMP_CHAINS_FILE"

# Clean up temp file
rm -f "$TEMP_CHAINS_FILE"

echo "📊 Found ${#CHAIN_ARRAY[@]} available protein chains"

# Randomly select chains for analysis
SEED=${RANDOM_SEED:-42}
RANDOM=$SEED

# Shuffle and select chains
TEMP_ARRAY=("${CHAIN_ARRAY[@]}")
for ((i=${#TEMP_ARRAY[@]}-1; i>0; i--)); do
    j=$((RANDOM % (i+1)))
    temp="${TEMP_ARRAY[i]}"
    TEMP_ARRAY[i]="${TEMP_ARRAY[j]}"
    TEMP_ARRAY[j]="$temp"
done

# Take more than we need to account for failed API calls
BUFFER_SIZE=$((TOTAL_SIZE + 50))
SELECTED_CHAINS=("${TEMP_ARRAY[@]:0:$BUFFER_SIZE}")

echo "🧬 Fetching PDB metadata for selected chains..."

# Function to get PDB information with retries and better error handling
get_pdb_info() {
    local pdb_chain="$1"
    local pdb_id=$(echo "$pdb_chain" | cut -d'_' -f1)
    local chain_id=$(echo "$pdb_chain" | cut -d'_' -f2)
    local max_retries=3
    local retry_delay=1

    for ((attempt=1; attempt<=max_retries; attempt++)); do
        # Fetch PDB summary information with timeout
        local api_url="https://data.rcsb.org/rest/v1/core/entry/${pdb_id}"
        local response=$(timeout 10 curl -s -f --connect-timeout 5 --max-time 10 "$api_url" 2>/dev/null)

        if [ $? -eq 0 ] && [ -n "$response" ] && [[ "$response" != *"error"* ]]; then
            # Extract key information using basic text processing
            local title=$(echo "$response" | grep -o '"title":"[^"]*"' | head -1 | sed 's/"title":"//; s/"//g' | sed 's/\\//g')
            local classification=$(echo "$response" | grep -o '"struct_keywords":"[^"]*"' | head -1 | sed 's/"struct_keywords":"//; s/"//g')
            local resolution=$(echo "$response" | grep -o '"resolution":[0-9.]*' | head -1 | sed 's/"resolution"://')
            local method=$(echo "$response" | grep -o '"experimental_method":\["[^"]*"' | head -1 | sed 's/"experimental_method":\["//; s/"//g')
            local organism=$(echo "$response" | grep -o '"source_organism":\["[^"]*"' | head -1 | sed 's/"source_organism":\["//; s/"//g')

            # Get chain-specific information with timeout
            local chain_api_url="https://data.rcsb.org/rest/v1/core/polymer_entity_instance/${pdb_id}/${chain_id}"
            local chain_response=$(timeout 10 curl -s -f --connect-timeout 5 --max-time 10 "$chain_api_url" 2>/dev/null)

            local length="null"
            local entity_type="Unknown"
            if [ $? -eq 0 ] && [ -n "$chain_response" ] && [[ "$chain_response" != *"error"* ]]; then
                length=$(echo "$chain_response" | grep -o '"length":[0-9]*' | head -1 | sed 's/"length"://')
                entity_type=$(echo "$chain_response" | grep -o '"type":"[^"]*"' | head -1 | sed 's/"type":"//; s/"//g')
            fi

            # Default values if not found
            title=${title:-"Unknown"}
            classification=${classification:-"Unknown"}
            resolution=${resolution:-"null"}
            method=${method:-"Unknown"}
            organism=${organism:-"Unknown"}
            length=${length:-"null"}
            entity_type=${entity_type:-"Unknown"}

            # Clean up any remaining problematic characters
            title=$(echo "$title" | tr -d '\n\r' | sed 's/[[:cntrl:]]//g')
            classification=$(echo "$classification" | tr -d '\n\r' | sed 's/[[:cntrl:]]//g')
            method=$(echo "$method" | tr -d '\n\r' | sed 's/[[:cntrl:]]//g')
            organism=$(echo "$organism" | tr -d '\n\r' | sed 's/[[:cntrl:]]//g')
            entity_type=$(echo "$entity_type" | tr -d '\n\r' | sed 's/[[:cntrl:]]//g')

            # Return JSON object
            cat << EOF
{
  "pdb_chain": "$pdb_chain",
  "pdb_id": "$pdb_id",
  "chain_id": "$chain_id",
  "title": "$title",
  "classification": "$classification",
  "resolution": $resolution,
  "experimental_method": "$method",
  "source_organism": "$organism",
  "chain_length": $length,
  "entity_type": "$entity_type"
}
EOF
            return 0
        else
            if [ $attempt -lt $max_retries ]; then
                sleep $retry_delay
                retry_delay=$((retry_delay * 2))
            fi
        fi
    done

    return 1
}

# Collect protein information with better progress tracking
PROTEIN_INFO=()
SUCCESSFUL_COUNT=0
FAILED_COUNT=0
BATCH_SIZE=10

echo "🔄 Processing protein metadata (this may take a few minutes)..."
echo "Target: $TOTAL_SIZE proteins, Buffer: $BUFFER_SIZE selected"

for ((i=0; i<${#SELECTED_CHAINS[@]}; i++)); do
    if [ $SUCCESSFUL_COUNT -ge $TOTAL_SIZE ]; then
        break
    fi

    chain="${SELECTED_CHAINS[i]}"

    # Progress indicator every 10 proteins
    if [ $((i % BATCH_SIZE)) -eq 0 ]; then
        echo "Progress: $i/${#SELECTED_CHAINS[@]} processed, $SUCCESSFUL_COUNT successful, $FAILED_COUNT failed"
    fi

    info=$(get_pdb_info "$chain")
    if [ $? -eq 0 ]; then
        PROTEIN_INFO+=("$info")
        ((SUCCESSFUL_COUNT++))
        echo -n "✓"
    else
        ((FAILED_COUNT++))
        echo -n "✗"
    fi

    # Rate limiting - be more conservative
    sleep 0.5

    # If we're failing too much, give a longer break
    if [ $((FAILED_COUNT % 10)) -eq 0 ] && [ $FAILED_COUNT -gt 0 ]; then
        echo ""
        echo "Taking longer break due to failures..."
        sleep 3
    fi
done

echo ""
echo "✅ Successfully fetched metadata for $SUCCESSFUL_COUNT proteins"
echo "❌ Failed to fetch metadata for $FAILED_COUNT proteins"

if [ $SUCCESSFUL_COUNT -lt $TOTAL_SIZE ]; then
    echo "⚠️  Warning: Only got data for $SUCCESSFUL_COUNT proteins, need $TOTAL_SIZE"
    echo "   Adjusting split sizes proportionally..."
    TRAIN_SIZE=$((SUCCESSFUL_COUNT * 80 / 130))
    VAL_SIZE=$((SUCCESSFUL_COUNT * 25 / 130))
    TEST_SIZE=$((SUCCESSFUL_COUNT - TRAIN_SIZE - VAL_SIZE))
    echo "   New sizes: Train=$TRAIN_SIZE, Val=$VAL_SIZE, Test=$TEST_SIZE"
fi

# === Create balanced splits based on biological features ===
echo "🧬 Creating biologically balanced splits..."

# Create temporary file for processing
TEMP_JSON=$(mktemp)
printf '[%s]' "$(IFS=','; echo "${PROTEIN_INFO[*]}")" > "$TEMP_JSON"

# Use Python for sophisticated balanced splitting
python3 << EOF
import json
import sys
import random
from collections import defaultdict, Counter
import tempfile
import os

# Load data from environment or create temp file
try:
    # Create temporary file with the protein data
    temp_file = '$TEMP_JSON'

    # Read protein info from bash array (passed as individual arguments)
    protein_data = []
EOF

# Pass the protein info to Python more reliably
cat > "$TEMP_JSON" << 'JSON_START'
[
JSON_START

# Add each protein info JSON, separated by commas
for ((i=0; i<${#PROTEIN_INFO[@]}; i++)); do
    if [ $i -gt 0 ]; then
        echo "," >> "$TEMP_JSON"
    fi
    echo "${PROTEIN_INFO[i]}" >> "$TEMP_JSON"
done

cat >> "$TEMP_JSON" << 'JSON_END'
]
JSON_END

# Continue with Python script
python3 << EOF
import json
import sys
import random
from collections import defaultdict, Counter
import tempfile

# Load data
with open(sys.argv[1], 'r') as f:
    proteins = json.load(f)

# Set random seed
random.seed(42)

# Define balancing features
def get_balance_features(protein):
    features = {}

    # Classification categories
    classification = protein.get('classification', '').lower()
    if 'enzyme' in classification or 'transferase' in classification or 'kinase' in classification:
        features['func_class'] = 'enzyme'
    elif 'binding' in classification or 'transport' in classification:
        features['func_class'] = 'binding'
    elif 'structural' in classification or 'fibrous' in classification:
        features['func_class'] = 'structural'
    elif 'immune' in classification or 'antibody' in classification:
        features['func_class'] = 'immune'
    else:
        features['func_class'] = 'other'

    # Size categories
    length = protein.get('chain_length')
    if length and length != 'null':
        if length < 100:
            features['size_class'] = 'small'
        elif length < 300:
            features['size_class'] = 'medium'
        elif length < 600:
            features['size_class'] = 'large'
        else:
            features['size_class'] = 'very_large'
    else:
        features['size_class'] = 'unknown'

    # Resolution categories
    resolution = protein.get('resolution')
    if resolution and resolution != 'null':
        if resolution < 1.5:
            features['resolution_class'] = 'high'
        elif resolution < 2.5:
            features['resolution_class'] = 'medium'
        else:
            features['resolution_class'] = 'low'
    else:
        features['resolution_class'] = 'unknown'

    # Experimental method
    method = protein.get('experimental_method', '').lower()
    if 'x-ray' in method:
        features['method_class'] = 'xray'
    elif 'nmr' in method:
        features['method_class'] = 'nmr'
    elif 'cryo' in method or 'electron' in method:
        features['method_class'] = 'cryo_em'
    else:
        features['method_class'] = 'other'

    # Organism type (simplified)
    organism = protein.get('source_organism', '').lower()
    if 'homo sapiens' in organism or 'human' in organism:
        features['organism_class'] = 'human'
    elif 'escherichia coli' in organism or 'e. coli' in organism:
        features['organism_class'] = 'ecoli'
    elif 'mouse' in organism or 'mus musculus' in organism:
        features['organism_class'] = 'mouse'
    elif 'bacteria' in organism or 'bacillus' in organism:
        features['organism_class'] = 'bacteria'
    else:
        features['organism_class'] = 'other'

    return features

# Add balance features to each protein
for protein in proteins:
    protein['balance_features'] = get_balance_features(protein)

# Stratified sampling for balanced splits
def create_balanced_splits(proteins, train_size=80, val_size=25, test_size=25):
    # Create strata based on combination of key features
    strata = defaultdict(list)

    for protein in proteins:
        features = protein['balance_features']
        # Create stratum key from most important features
        stratum_key = (
            features['func_class'],
            features['size_class'],
            features['method_class']
        )
        strata[stratum_key].append(protein)

    # Shuffle proteins within each stratum
    for stratum in strata.values():
        random.shuffle(stratum)

    train_set = []
    val_set = []
    test_set = []

    # Distribute from each stratum proportionally
    for stratum_proteins in strata.values():
        n_proteins = len(stratum_proteins)
        n_train = int(n_proteins * train_size / (train_size + val_size + test_size))
        n_val = int(n_proteins * val_size / (train_size + val_size + test_size))
        n_test = n_proteins - n_train - n_val

        train_set.extend(stratum_proteins[:n_train])
        val_set.extend(stratum_proteins[n_train:n_train + n_val])
        test_set.extend(stratum_proteins[n_train + n_val:n_train + n_val + n_test])

    # Final shuffle of each set
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    # Adjust sizes if needed
    all_sets = [train_set, val_set, test_set]
    target_sizes = [train_size, val_size, test_size]

    # Simple rebalancing
    while sum(len(s) for s in all_sets) > train_size + val_size + test_size:
        # Remove from largest set
        largest_idx = max(range(3), key=lambda i: len(all_sets[i]) - target_sizes[i])
        if len(all_sets[largest_idx]) > target_sizes[largest_idx]:
            all_sets[largest_idx].pop()

    while sum(len(s) for s in all_sets) < train_size + val_size + test_size:
        # Add to smallest set from remaining proteins
        smallest_idx = min(range(3), key=lambda i: len(all_sets[i]) - target_sizes[i])
        # This is simplified - in practice you'd pull from unused proteins
        break

    return all_sets[0][:train_size], all_sets[1][:val_size], all_sets[2][:test_size]

# Create splits
train_proteins, val_proteins, test_proteins = create_balanced_splits(proteins)

# Generate summary statistics
def analyze_split(split_proteins, split_name):
    analysis = {
        'count': len(split_proteins),
        'feature_distribution': {}
    }

    for feature_type in ['func_class', 'size_class', 'resolution_class', 'method_class', 'organism_class']:
        distribution = Counter(p['balance_features'][feature_type] for p in split_proteins)
        analysis['feature_distribution'][feature_type] = dict(distribution)

    return analysis

# Create final output
output = {
    'metadata': {
        'total_proteins': len(proteins),
        'random_seed': 42,
        'split_strategy': 'stratified_biological_features',
        'balance_features': [
            'functional_classification',
            'protein_size',
            'resolution_quality',
            'experimental_method',
            'source_organism'
        ]
    },
    'splits': {
        'training': {
            'proteins': train_proteins,
            'analysis': analyze_split(train_proteins, 'training')
        },
        'validation': {
            'proteins': val_proteins,
            'analysis': analyze_split(val_proteins, 'validation')
        },
        'testing': {
            'proteins': test_proteins,
            'analysis': analyze_split(test_proteins, 'testing')
        }
    }
}

# Save to file
output_file = sys.argv[2]
with open(output_file, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Balanced splits saved to {output_file}")

# Print summary
print("\n📊 Split Analysis:")
for split_name, split_data in output['splits'].items():
    print(f"\n{split_name.upper()} SET ({split_data['analysis']['count']} proteins):")
    for feature, distribution in split_data['analysis']['feature_distribution'].items():
        print(f"  {feature}: {distribution}")

EOF

# Run Python script
OUTPUT_FILE="${BASE_DIR}/balanced_protein_splits.json"
python3 - "$TEMP_JSON" "$OUTPUT_FILE" << 'EOF'
# (Python script is embedded above)
EOF

# Clean up
rm -f "$TEMP_JSON"

echo ""
echo "🎯 Balanced dataset splits created successfully!"
echo "📄 Output file: $OUTPUT_FILE"
echo ""
echo "📋 The JSON contains:"
echo "   - Detailed protein metadata from PDB"
echo "   - Biological balance features used for splitting"
echo "   - Training/validation/testing assignments"
echo "   - Distribution analysis for each split"
echo ""
echo "🔬 Balance features used:"
echo "   - Functional classification (enzyme, binding, structural, immune, other)"
echo "   - Protein size (small <100, medium 100-300, large 300-600, very_large >600)"
echo "   - Resolution quality (high <1.5Å, medium 1.5-2.5Å, low >2.5Å)"
echo "   - Experimental method (X-ray, NMR, Cryo-EM, other)"
echo "   - Source organism (human, E.coli, mouse, bacteria, other)"
EOF