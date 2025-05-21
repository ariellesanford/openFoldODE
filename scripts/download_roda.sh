#!/bin/bash
#DONT RUN UNLESS YOU WANT EVERYTHING IN RODA!
set -e

# Define target base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"  # Move up one level to project root
BASE_DIR="${ROOT_DIR}/neural_ODE/data/RODA"

# Define paths
ALIGNMENT_DIR="$BASE_DIR/alignment_data/alignment_dir_roda"
PDB_DIR="$BASE_DIR/pdb_data"
CHAIN_FILE="$BASE_DIR/duplicate_pdb_chains.txt"

echo "📁 Creating necessary directories in $BASE_DIR..."
mkdir -p "$ALIGNMENT_DIR"
mkdir -p "$PDB_DIR"

echo "☁️ Downloading alignment files from OpenFold S3..."
aws s3 cp s3://openfold/pdb/ "$ALIGNMENT_DIR" --recursive --no-sign-request

echo "☁️ Downloading mmCIF structures and chain duplicates file..."
aws s3 cp s3://openfold/pdb_mmcif.zip "$PDB_DIR/pdb_mmcif.zip" --no-sign-request
aws s3 cp s3://openfold/duplicate_pdb_chains.txt "$CHAIN_FILE" --no-sign-request

echo "📦 Unzipping mmCIF structure files into $PDB_DIR..."
unzip -q "$PDB_DIR/pdb_mmcif.zip" -d "$PDB_DIR"

echo "✅ All data downloaded and unpacked into:"
echo "  → Alignments: $ALIGNMENT_DIR"
echo "  → Structures: $PDB_DIR"
echo "  → Duplicate chains list: $CHAIN_FILE"
