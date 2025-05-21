#!/bin/bash
set -e

# === CONFIG ===
DATA_DIR="/home/visitor/Desktop/homoSapien_humoralImmuneResponse"
SEQUENCE_DIR="${DATA_DIR}/training_data(75%)/sequences"
ALIGNMENT_CLUSTER_DIR="${DATA_DIR}/training_data(75%)/alignment_data"
CONCAT_FASTA="${ALIGNMENT_CLUSTER_DIR}/all_training_seqs.fasta"
CLUSTER_OUTPUT="${ALIGNMENT_CLUSTER_DIR}/clustered_seqs"
CLUSTER_TXT="${ALIGNMENT_CLUSTER_DIR}/all-seqs_clusters-40.txt"

# === Check dependencies ===
if ! command -v cd-hit &> /dev/null; then
    echo "❌ cd-hit is not installed. Please install it with: sudo apt install cd-hit"
    exit 1
fi

# === Step 2: Concatenate all FASTA files ===
echo "📎 Concatenating FASTA files into: $CONCAT_FASTA"
cat "$SEQUENCE_DIR"/*.fasta > "$CONCAT_FASTA"
echo "✅ Concatenation complete."

# === Step 3: Run CD-HIT at 40% sequence identity ===
echo "🔬 Running CD-HIT clustering..."
cd-hit -i "$CONCAT_FASTA" -o "$CLUSTER_OUTPUT" -c 0.4 -n 2
echo "✅ Clustering complete. Output: ${CLUSTER_OUTPUT}.clstr"

# === Step 4: Parse .clstr file into OpenFold cluster format ===
echo "🧠 Generating cluster mapping: $CLUSTER_TXT"
awk '
  BEGIN {i=0}
  /^>/ {cluster="cluster" ++i; next}
  {
    match($0, />?([^ ]+)/, arr);
    gsub(/\*/,"", arr[1]);
    print arr[1], cluster
  }
' "${CLUSTER_OUTPUT}.clstr" > "$CLUSTER_TXT"
echo "✅ Cluster mapping written to: $CLUSTER_TXT"

# === Done ===
echo "🎉 Sequence clustering pipeline complete."
