#!/bin/bash
set -e

# Usage: bash lookup_pdb_chain_metadata.sh 1lfh_A

if [ $# -ne 1 ]; then
  echo "Usage: $0 <PDB_CHAIN>  (format: XXXX_A)"
  exit 1
fi

PDB_CHAIN=$(echo "$1" | tr '[:upper:]' '[:lower:]')
PDB_ID=$(echo "$PDB_CHAIN" | cut -d'_' -f1)
CHAIN_ID=$(echo "$PDB_CHAIN" | cut -d'_' -f2)

ENTRY_JSON=$(env -i PATH=/usr/bin:/bin /usr/bin/curl -s "https://data.rcsb.org/rest/v1/core/entry/${PDB_ID}")
ENTITY_IDS=$(echo "$ENTRY_JSON" | jq -r '.rcsb_entry_container_identifiers.polymer_entity_ids[]')

FOUND=false
for ENTITY_ID in $ENTITY_IDS; do
  ENTITY_JSON=$(env -i PATH=/usr/bin:/bin /usr/bin/curl -s "https://data.rcsb.org/rest/v1/core/polymer_entity/${PDB_ID}/${ENTITY_ID}")
  CHAINS=$(echo "$ENTITY_JSON" | jq -r '.rcsb_polymer_entity_container_identifiers.auth_asym_ids[]')

  for C in $CHAINS; do
    if [[ "${C,,}" == "$CHAIN_ID" ]]; then
      echo "$ENTITY_JSON" | jq '{
        pdb_id: .rcsb_polymer_entity_container_identifiers.entry_id,
        entity_id: .rcsb_polymer_entity_container_identifiers.entity_id,
        chain_ids: .rcsb_polymer_entity_container_identifiers.auth_asym_ids,
        molecule_name: .rcsb_polymer_entity.pdbx_description,
        sequence_length: (.entity_poly.pdbx_seq_one_letter_code_can | gsub("[\n ]"; "") | length),
        entity_type: .entity_poly.type,
        ec_numbers: (.rcsb_polymer_entity.rcsb_ec_lineage // []),
        organism: (.rcsb_entity_source_organism[0].scientific_name // "Unknown"),
        taxonomy_id: (.rcsb_entity_source_organism[0].ncbi_taxonomy_id // "N/A")
      }'
      FOUND=true
      break 2
    fi
  done
done

if ! $FOUND; then
  echo "❌ Chain ${CHAIN_ID^^} not found in polymer entities of ${PDB_ID^^}"
  exit 1
fi
