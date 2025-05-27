#!/bin/bash
#
# Test metadata collection with verbose debugging
# Usage: bash test_metadata_collection.sh [chain_id]

set -e  # Remove this to continue on errors for debugging

CHAIN=${1:-"1wt0_A"}
echo "🧪 Testing metadata collection for chain: $CHAIN"
echo ""

# Enable verbose debugging
set -x

get_chain_metadata_debug() {
    local chain="$1"
    local pdb_code=$(echo "$chain" | cut -d'_' -f1)
    local chain_id="A"

    echo "=== DEBUG: Processing chain: $chain (PDB: $pdb_code, Chain: $chain_id) ===" >&2

    # Get entry information
    echo "=== DEBUG: Fetching entry information ===" >&2
    local entry_json
    entry_json=$(env -i PATH=/usr/bin:/bin /usr/bin/curl -s -f "https://data.rcsb.org/rest/v1/core/entry/${pdb_code}" 2>&1)
    local curl_exit_code=$?

    echo "=== DEBUG: Curl exit code: $curl_exit_code ===" >&2
    echo "=== DEBUG: Entry JSON length: ${#entry_json} ===" >&2

    if [ $curl_exit_code -ne 0 ]; then
        echo "=== DEBUG: Curl failed with exit code $curl_exit_code ===" >&2
        echo "=== DEBUG: Curl output: $entry_json ===" >&2
        return 1
    fi

    if [ -z "$entry_json" ]; then
        echo "=== DEBUG: Entry JSON is empty ===" >&2
        return 1
    fi

    # Check if jq is working
    echo "=== DEBUG: Testing jq on entry JSON ===" >&2
    local test_jq
    test_jq=$(echo "$entry_json" | jq -r '.rcsb_id // "NOT_FOUND"' 2>&1)
    local jq_exit_code=$?
    echo "=== DEBUG: jq test result: $test_jq (exit code: $jq_exit_code) ===" >&2

    # Get polymer entity IDs
    echo "=== DEBUG: Extracting entity IDs ===" >&2
    local entity_ids
    entity_ids=$(echo "$entry_json" | jq -r '.rcsb_entry_container_identifiers.polymer_entity_ids[]?' 2>&1)
    local jq_entity_exit=$?

    echo "=== DEBUG: jq entity extraction exit code: $jq_entity_exit ===" >&2
    echo "=== DEBUG: Entity IDs: '$entity_ids' ===" >&2

    if [ $jq_entity_exit -ne 0 ]; then
        echo "=== DEBUG: jq failed to extract entity IDs ===" >&2
        echo "=== DEBUG: jq error: $entity_ids ===" >&2
        return 1
    fi

    if [ -z "$entity_ids" ]; then
        echo "=== DEBUG: No entity IDs found ===" >&2
        # Let's see what the JSON structure actually looks like
        echo "=== DEBUG: Full JSON structure: ===" >&2
        echo "$entry_json" | jq . 2>&1 >&2 || echo "JSON parse failed" >&2
        return 1
    fi

    # Process each entity
    local entity_count=0
    for entity_id in $entity_ids; do
        entity_count=$((entity_count + 1))
        echo "=== DEBUG: Processing entity $entity_count: $entity_id ===" >&2

        local entity_json
        entity_json=$(env -i PATH=/usr/bin:/bin /usr/bin/curl -s -f "https://data.rcsb.org/rest/v1/core/polymer_entity/${pdb_code}/${entity_id}" 2>&1)
        local entity_curl_exit=$?

        echo "=== DEBUG: Entity curl exit code: $entity_curl_exit ===" >&2
        echo "=== DEBUG: Entity JSON length: ${#entity_json} ===" >&2

        if [ $entity_curl_exit -ne 0 ] || [ -z "$entity_json" ]; then
            echo "=== DEBUG: Failed to get entity $entity_id, skipping ===" >&2
            continue
        fi

        local chains
        chains=$(echo "$entity_json" | jq -r '.rcsb_polymer_entity_container_identifiers.auth_asym_ids[]?' 2>&1)
        local chains_jq_exit=$?

        echo "=== DEBUG: Chains jq exit code: $chains_jq_exit ===" >&2
        echo "=== DEBUG: Chains found: '$chains' ===" >&2

        if [ $chains_jq_exit -ne 0 ]; then
            echo "=== DEBUG: Failed to extract chains, skipping entity ===" >&2
            continue
        fi

        # Check if this entity contains chain A
        local chain_found=false
        for c in $chains; do
            echo "=== DEBUG: Checking chain '$c' against 'a' ===" >&2
            if [[ "${c,,}" == "a" ]]; then
                echo "=== DEBUG: Found matching chain A! ===" >&2
                chain_found=true

                # Extract metadata
                local molecule_name
                molecule_name=$(echo "$entity_json" | jq -r '.rcsb_polymer_entity.pdbx_description // "Unknown"' 2>/dev/null)
                echo "=== DEBUG: Molecule name: '$molecule_name' ===" >&2

                local sequence_length
                sequence_length=$(echo "$entity_json" | jq -r '(.entity_poly.pdbx_seq_one_letter_code_can | gsub("[\n ]"; "") | length) // 0' 2>/dev/null)
                echo "=== DEBUG: Sequence length: '$sequence_length' ===" >&2

                # Output basic JSON
                cat << EOJ
{
  "pdb_chain": "$chain",
  "pdb_id": "$pdb_code",
  "chain_id": "A",
  "entity_id": "$entity_id",
  "molecule_name": "$(echo "$molecule_name" | sed 's/"/\\"/g')",
  "sequence_length": $sequence_length,
  "status": "success"
}
EOJ
                return 0
            fi
        done

        if [ "$chain_found" = false ]; then
            echo "=== DEBUG: Chain A not found in entity $entity_id ===" >&2
        fi
    done

    echo "=== DEBUG: Chain A not found in any entity ===" >&2
    return 1
}

# Test the function
echo "Running metadata collection test..."
echo ""

result=$(get_chain_metadata_debug "$CHAIN" 2>&1)
exit_code=$?

echo ""
echo "=== RESULT ==="
echo "Exit code: $exit_code"
echo "Output:"
echo "$result"

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Metadata collection successful!"
else
    echo ""
    echo "❌ Metadata collection failed!"
fi