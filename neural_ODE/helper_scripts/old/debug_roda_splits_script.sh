#!/bin/bash
#
# Debug version with verbose logging
# Usage: bash debug_fixed_balanced_splits.sh

# Don't exit on errors for debugging
# set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR="${SCRIPT_DIR}"

# Just test with 3 chains for debugging
TEST_CHAINS=("1wt0_A" "1abc_A" "2abc_A")
TEMP_DIR=$(mktemp -d)

echo "🧪 Debug version - testing with ${#TEST_CHAINS[@]} chains"
echo "Temp directory: $TEMP_DIR"
echo ""

# Function to safely extract JSON values
safe_jq() {
    local json="$1"
    local query="$2"
    local default="$3"

    echo "=== safe_jq DEBUG ===" >&2
    echo "Query: $query" >&2
    echo "JSON length: ${#json}" >&2

    # First check if JSON is valid
    if ! echo "$json" | jq . >/dev/null 2>&1; then
        echo "JSON validation failed" >&2
        echo "$default"
        return 1
    fi
    echo "JSON validation passed" >&2

    # Then extract the value
    local result
    result=$(echo "$json" | jq -r "$query" 2>&1)
    local jq_exit=$?
    echo "jq exit code: $jq_exit" >&2
    echo "jq result: '$result'" >&2

    if [[ "$result" == "null" || -z "$result" ]]; then
        echo "$default"
    else
        echo "$result"
    fi
}

# Function to get metadata for a single chain
get_chain_metadata_debug() {
    local chain="$1"
    local pdb_code=$(echo "$chain" | cut -d'_' -f1)
    local chain_id="A"

    echo ""
    echo "=== PROCESSING CHAIN: $chain ===" >&2
    echo "PDB code: $pdb_code" >&2
    echo "Chain ID: $chain_id" >&2

    # Test which curl works
    echo "Testing curl commands..." >&2

    # Test 1: system curl
    echo "Test 1: System curl" >&2
    local test1_response
    test1_response=$(env -i PATH=/usr/bin:/bin /usr/bin/curl -s -w "HTTPCODE:%{http_code}" "https://data.rcsb.org/rest/v1/core/entry/${pdb_code}" 2>&1)
    local test1_exit=$?
    echo "System curl exit: $test1_exit" >&2
    echo "System curl response length: ${#test1_response}" >&2
    local test1_http=$(echo "$test1_response" | grep -o "HTTPCODE:[0-9]*" | cut -d: -f2)
    echo "System curl HTTP code: $test1_http" >&2

    # Test 2: conda curl
    echo "Test 2: Conda curl" >&2
    local test2_response
    test2_response=$(curl -s -w "HTTPCODE:%{http_code}" "https://data.rcsb.org/rest/v1/core/entry/${pdb_code}" 2>&1)
    local test2_exit=$?
    echo "Conda curl exit: $test2_exit" >&2
    echo "Conda curl response length: ${#test2_response}" >&2
    local test2_http=$(echo "$test2_response" | grep -o "HTTPCODE:[0-9]*" | cut -d: -f2)
    echo "Conda curl HTTP code: $test2_http" >&2

    # Choose the better response
    local entry_response
    local entry_http_code
    local entry_json

    if [[ "$test1_http" == "200" ]]; then
        echo "Using system curl response" >&2
        entry_response="$test1_response"
        entry_http_code="$test1_http"
        entry_json=$(echo "$entry_response" | sed 's/HTTPCODE:[0-9]*$//')
    elif [[ "$test2_http" == "200" ]]; then
        echo "Using conda curl response" >&2
        entry_response="$test2_response"
        entry_http_code="$test2_http"
        entry_json=$(echo "$entry_response" | sed 's/HTTPCODE:[0-9]*$//')
    else
        echo "Both curl commands failed!" >&2
        echo "System curl response: $test1_response" >&2
        echo "Conda curl response: $test2_response" >&2
        return 1
    fi

    echo "Selected HTTP code: $entry_http_code" >&2
    echo "Selected JSON length: ${#entry_json}" >&2

    if [[ "$entry_http_code" != "200" ]] || [[ -z "$entry_json" ]]; then
        echo "HTTP error or empty response" >&2
        return 1
    fi

    # Show first part of JSON for debugging
    echo "JSON preview:" >&2
    echo "$entry_json" | head -c 200 >&2
    echo "..." >&2

    # Validate JSON
    echo "Validating JSON..." >&2
    if ! echo "$entry_json" | jq . >/dev/null 2>&1; then
        echo "JSON validation failed!" >&2
        echo "Invalid JSON content:" >&2
        echo "$entry_json" | head -c 500 >&2
        return 1
    fi
    echo "JSON validation passed" >&2

    # Get polymer entity IDs
    echo "Extracting entity IDs..." >&2
    local entity_ids
    entity_ids=$(safe_jq "$entry_json" '.rcsb_entry_container_identifiers.polymer_entity_ids[]?' "")
    echo "Entity IDs result: '$entity_ids'" >&2

    if [[ -z "$entity_ids" ]]; then
        echo "No entity IDs found!" >&2
        # Show the JSON structure to debug
        echo "JSON structure around entity IDs:" >&2
        echo "$entry_json" | jq '.rcsb_entry_container_identifiers' 2>&1 >&2
        return 1
    fi

    echo "Found entity IDs: $entity_ids" >&2

    # Process first entity only for debugging
    local first_entity=$(echo "$entity_ids" | head -1)
    echo "Processing first entity: $first_entity" >&2

    # Get entity info
    echo "Fetching entity information..." >&2
    local entity_response
    entity_response=$(env -i PATH=/usr/bin:/bin /usr/bin/curl -s -w "HTTPCODE:%{http_code}" "https://data.rcsb.org/rest/v1/core/polymer_entity/${pdb_code}/${first_entity}" 2>&1)
    local entity_http_code=$(echo "$entity_response" | grep -o "HTTPCODE:[0-9]*" | cut -d: -f2)
    local entity_json=$(echo "$entity_response" | sed 's/HTTPCODE:[0-9]*$//')

    echo "Entity HTTP code: $entity_http_code" >&2
    echo "Entity JSON length: ${#entity_json}" >&2

    if [[ "$entity_http_code" != "200" ]] || [[ -z "$entity_json" ]]; then
        echo "Entity request failed!" >&2
        return 1
    fi

    # Validate entity JSON
    if ! echo "$entity_json" | jq . >/dev/null 2>&1; then
        echo "Entity JSON validation failed!" >&2
        return 1
    fi

    echo "Entity JSON validation passed" >&2

    # Get chains
    local chains
    chains=$(safe_jq "$entity_json" '.rcsb_polymer_entity_container_identifiers.auth_asym_ids[]?' "")
    echo "Chains found: '$chains'" >&2

    # Check for chain A
    local found_chain_a=false
    for c in $chains; do
        echo "Checking chain: '$c'" >&2
        if [[ "${c,,}" == "a" ]]; then
            echo "Found chain A!" >&2
            found_chain_a=true
            break
        fi
    done

    if [[ "$found_chain_a" == "false" ]]; then
        echo "Chain A not found in this entity" >&2
        return 1
    fi

    # Extract basic metadata
    echo "Extracting metadata..." >&2
    local molecule_name=$(safe_jq "$entity_json" '.rcsb_polymer_entity.pdbx_description' "Unknown")
    local sequence_length=$(safe_jq "$entity_json" '(.entity_poly.pdbx_seq_one_letter_code_can | gsub("[\n ]"; "") | length)' "0")

    echo "Molecule name: '$molecule_name'" >&2
    echo "Sequence length: '$sequence_length'" >&2

    # Create simple JSON output
    cat << EOF
{
  "pdb_chain": "$chain",
  "pdb_id": "$pdb_code",
  "chain_id": "A",
  "entity_id": "$first_entity",
  "molecule_name": "$(echo "$molecule_name" | sed 's/"/\\"/g')",
  "sequence_length": $sequence_length,
  "status": "debug_success"
}
EOF

    echo "Metadata extraction completed successfully!" >&2
    return 0
}

# Test with each chain
for i in "${!TEST_CHAINS[@]}"; do
    chain="${TEST_CHAINS[$i]}"
    echo ""
    echo "==================== TESTING CHAIN $((i+1))/${#TEST_CHAINS[@]}: $chain ===================="

    result=$(get_chain_metadata_debug "$chain" 2>&1)
    exit_code=$?

    echo ""
    echo "=== RESULT FOR $chain ==="
    echo "Exit code: $exit_code"
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS"
        echo "JSON output:"
        echo "$result" | grep -A 20 '^{'
    else
        echo "❌ FAILED"
    fi
    echo ""
    echo "Full debug output:"
    echo "$result"
    echo ""

    # Add delay between requests
    sleep 2
done

echo "Debug test completed!"
echo "Temp directory: $TEMP_DIR (not cleaned up for inspection)"