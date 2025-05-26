#!/bin/bash
# Debug version to diagnose issues with RODA chain discovery

set -e

echo "🔍 Starting debug diagnostics..."
echo ""

# Check AWS CLI
echo "1. Checking AWS CLI installation:"
which aws || echo "   ❌ AWS CLI not found in PATH"
aws --version 2>/dev/null || echo "   ❌ Cannot get AWS version"
echo ""

# Test S3 listing
echo "2. Testing S3 bucket access:"
echo "   Running: aws s3 ls s3://openfold/pdb/ --no-sign-request | head -20"
echo "   Output:"
aws s3 ls s3://openfold/pdb/ --no-sign-request 2>&1 | head -20 || echo "   ❌ S3 listing failed"
echo ""

# Check what the grep/awk pipeline produces
echo "3. Testing chain extraction pipeline:"
TEMP_FILE=$(mktemp)
echo "   Saving first 50 entries to temp file..."
aws s3 ls s3://openfold/pdb/ --no-sign-request 2>/dev/null | grep 'PRE' | awk '{print $2}' | sed 's/\///g' | head -50 > "$TEMP_FILE" || {
    echo "   ❌ Pipeline failed"
    echo "   Trying without grep PRE filter:"
    aws s3 ls s3://openfold/pdb/ --no-sign-request 2>/dev/null | head -20
}

echo "   First 10 extracted chains:"
head -10 "$TEMP_FILE" 2>/dev/null || echo "   ❌ No chains extracted"
echo ""

echo "4. Checking array population:"
CHAIN_ARRAY=()
while IFS= read -r line; do
    if [[ -n "$line" ]]; then
        CHAIN_ARRAY+=("$line")
    fi
done < "$TEMP_FILE"

echo "   Array size: ${#CHAIN_ARRAY[@]}"
echo "   First 5 array elements:"
for i in {0..4}; do
    echo "     [$i]: ${CHAIN_ARRAY[$i]:-EMPTY}"
done
echo ""

# Check Python availability
echo "5. Checking Python 3:"
which python3 || echo "   ❌ python3 not found"
python3 --version 2>/dev/null || echo "   ❌ Cannot get python3 version"
echo ""

# Check curl
echo "6. Checking curl:"
which curl || echo "   ❌ curl not found"
curl --version | head -1 2>/dev/null || echo "   ❌ Cannot get curl version"
echo ""

# Test PDB API
echo "7. Testing PDB API access:"
echo "   Fetching info for test PDB entry 1a8o..."
curl -s -f "https://data.rcsb.org/rest/v1/core/entry/1a8o" 2>&1 | head -100 | python3 -m json.tool 2>/dev/null || echo "   ❌ PDB API test failed"
echo ""

# Clean up
rm -f "$TEMP_FILE"

echo "✅ Diagnostics complete!"
echo ""
echo "If any of the above checks failed, please install the missing dependencies:"
echo "  - AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html"
echo "  - Python 3: sudo apt-get install python3"
echo "  - curl: sudo apt-get install curl"