#!/bin/bash
#
# Simplified test version to identify the issue
# Only processes 5 chains with verbose debugging

# DON'T exit on errors initially - we want to see what fails
# set -e

echo "🧪 Simple test version - processing only 5 chains"
echo ""

# Test chains (known to exist)
TEST_CHAINS=("1abc_A" "1wt0_A" "2abc_A" "1def_A" "3abc_A")

TEMP_DIR=$(mktemp -d)
echo "Using temp directory: $TEMP_DIR"

METADATA_FILE="${TEMP_DIR}/metadata.json"
echo "[" > "$METADATA_FILE"

SUCCESSFUL_COUNT=0
FAILED_COUNT=0
FIRST_ENTRY=true

echo "Starting metadata collection..."

for i in "${!TEST_CHAINS[@]}"; do
    chain="${TEST_CHAINS[$i]}"
    echo ""
    echo "=== Processing chain $((i+1))/5: $chain ==="

    pdb_code=$(echo "$chain" | cut -d'_' -f1)
    echo "PDB code: $pdb_code"

    # Test basic connectivity first
    echo "Testing basic connectivity..."
    test_url="https://data.rcsb.org/rest/v1/core/entry/${pdb_code}"
    echo "URL: $test_url"

    # Use curl with verbose error reporting
    echo "Making curl request..."
    curl_output=$(curl -s -w "HTTPCODE:%{http_code}" "$test_url" 2>&1)
    curl_exit=$?

    echo "Curl exit code: $curl_exit"
    echo "Curl output length: ${#curl_output}"

    if [ $curl_exit -ne 0 ]; then
        echo "❌ Curl failed for $chain"
        echo "Curl error output: $curl_output"
        ((FAILED_COUNT++))
        continue
    fi

    # Extract HTTP code
    http_code=$(echo "$curl_output" | grep -o "HTTPCODE:[0-9]*" | cut -d: -f2)
    json_content=$(echo "$curl_output" | sed 's/HTTPCODE:[0-9]*$//')

    echo "HTTP code: $http_code"
    echo "JSON content length: ${#json_content}"

    if [[ "$http_code" != "200" ]]; then
        echo "❌ HTTP error $http_code for $chain"
        ((FAILED_COUNT++))
        continue
    fi

    if [[ -z "$json_content" ]]; then
        echo "❌ Empty response for $chain"
        ((FAILED_COUNT++))
        continue
    fi

    # Test jq parsing
    echo "Testing jq parsing..."
    jq_test=$(echo "$json_content" | jq -r '.rcsb_id // "NOT_FOUND"' 2>&1)
    jq_exit=$?

    echo "jq test result: $jq_test (exit code: $jq_exit)"

    if [ $jq_exit -ne 0 ]; then
        echo "❌ jq parsing failed for $chain"
        echo "jq error: $jq_test"
        ((FAILED_COUNT++))
        continue
    fi

    # Try to get entity IDs
    echo "Extracting entity IDs..."
    entity_ids=$(echo "$json_content" | jq -r '.rcsb_entry_container_identifiers.polymer_entity_ids[]?' 2>&1)
    entity_jq_exit=$?

    echo "Entity extraction exit code: $entity_jq_exit"
    echo "Entity IDs: '$entity_ids'"

    if [ $entity_jq_exit -ne 0 ] || [ -z "$entity_ids" ]; then
        echo "❌ No entity IDs found for $chain"
        echo "Let's see the JSON structure:"
        echo "$json_content" | jq . 2>&1 | head -20
        ((FAILED_COUNT++))
        continue
    fi

    echo "✅ Successfully processed $chain"

    # Create a simple JSON entry
    if [ "$FIRST_ENTRY" = true ]; then
        FIRST_ENTRY=false
    else
        echo "," >> "$METADATA_FILE"
    fi

    cat >> "$METADATA_FILE" << EOF
  {
    "pdb_chain": "$chain",
    "pdb_id": "$pdb_code",
    "status": "success",
    "entity_ids": "$(echo "$entity_ids" | tr '\n' ' ' | sed 's/"/\\"/g')"
  }
EOF

    ((SUCCESSFUL_COUNT++))

    # Add delay to be nice to the API
    echo "Sleeping for 1 second..."
    sleep 1
done

echo "]" >> "$METADATA_FILE"

echo ""
echo "=== SUMMARY ==="
echo "Successful: $SUCCESSFUL_COUNT"
echo "Failed: $FAILED_COUNT"

if [ $SUCCESSFUL_COUNT -gt 0 ]; then
    echo ""
    echo "Generated metadata file:"
    cat "$METADATA_FILE"

    echo ""
    echo "Testing jq on generated file:"
    jq . "$METADATA_FILE" 2>&1 || echo "jq failed on generated file"
fi

echo ""
echo "Temp directory: $TEMP_DIR"
echo "Metadata file: $METADATA_FILE"
echo ""
echo "To clean up: rm -rf $TEMP_DIR"