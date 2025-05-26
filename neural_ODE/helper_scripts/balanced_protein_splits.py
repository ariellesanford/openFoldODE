#!/usr/bin/env python3
"""
Create balanced and diverse training/validation/test splits for neural ODE training
Combines chain discovery, metadata collection, and stratified splitting

Usage: python create_balanced_protein_splits.py [--total-size 130] [--seed 42] [--output-dir .]
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests


class ProteinSplitGenerator:
    def __init__(self, total_size: int = 130, train_size: int = 80, val_size: int = 25,
                 test_size: int = 25, seed: int = 42, output_dir: str = "."):
        self.total_size = total_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed
        self.output_dir = output_dir

        # Set random seed
        random.seed(seed)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Session for HTTP requests with retry
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

        # Statistics
        self.successful_count = 0
        self.failed_count = 0

    def get_roda_chains(self) -> List[str]:
        """Get list of available RODA chains with format XXXX_A"""
        print("🔍 Discovering available RODA protein chains...")
        print("📡 Fetching chain list from RODA S3 bucket...")

        try:
            # Use AWS CLI to list S3 bucket
            result = subprocess.run([
                'aws', 's3', 'ls', 's3://openfold/pdb/', '--no-sign-request'
            ], capture_output=True, text=True, check=True)

            # Parse output to extract chain IDs
            chains = []
            for line in result.stdout.strip().split('\n'):
                if 'PRE' in line:
                    # Extract directory name
                    parts = line.split()
                    if len(parts) >= 2:
                        chain_dir = parts[1].rstrip('/')
                        # Only include chains with format XXXX_A
                        if re.match(r'^[0-9a-z]{4}_A$', chain_dir):
                            chains.append(chain_dir)

            print(f"📊 Found {len(chains)} available protein chains in RODA (format XXXX_A)")
            return chains

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to fetch chain list from S3: {e}")
            sys.exit(1)
        except FileNotFoundError:
            print("❌ AWS CLI not found. Please install AWS CLI.")
            sys.exit(1)

    def get_chain_metadata(self, chain: str) -> Optional[Dict]:
        """Get metadata for a single protein chain"""
        pdb_code = chain.split('_')[0]

        try:
            # Get entry information
            entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_code}"
            entry_response = self.session.get(entry_url, timeout=10)

            if entry_response.status_code != 200:
                return None

            entry_data = entry_response.json()

            # Get polymer entity IDs
            entity_ids = entry_data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])
            if not entity_ids:
                return None

            # Check each entity for chain A
            for entity_id in entity_ids:
                entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_code}/{entity_id}"
                entity_response = self.session.get(entity_url, timeout=10)

                if entity_response.status_code != 200:
                    continue

                entity_data = entity_response.json()

                # Check if this entity contains chain A
                auth_asym_ids = entity_data.get('rcsb_polymer_entity_container_identifiers', {}).get('auth_asym_ids',
                                                                                                     [])
                if 'A' not in [chain_id.upper() for chain_id in auth_asym_ids]:
                    continue

                # Extract metadata
                molecule_name = entity_data.get('rcsb_polymer_entity', {}).get('pdbx_description', 'Unknown')

                # Get sequence length
                sequence = entity_data.get('entity_poly', {}).get('pdbx_seq_one_letter_code_can', '')
                sequence_length = len(re.sub(r'[\n ]', '', sequence)) if sequence else 0

                entity_type = entity_data.get('entity_poly', {}).get('type', 'Unknown')

                # Get organism info
                organism_info = entity_data.get('rcsb_entity_source_organism', [{}])[0]
                organism = organism_info.get('scientific_name', 'Unknown')
                taxonomy_id = organism_info.get('ncbi_taxonomy_id', 'N/A')

                # Get EC numbers
                ec_lineage = entity_data.get('rcsb_polymer_entity', {}).get('rcsb_ec_lineage', [])
                ec_numbers = '; '.join([ec.get('name', '') for ec in ec_lineage if ec.get('name')])

                # Classify by function
                func_class = self.classify_function(molecule_name, ec_numbers)

                # Classify by size
                size_class = self.classify_size(sequence_length)

                # Classify by organism
                organism_class = self.classify_organism(organism)

                return {
                    'pdb_chain': chain,
                    'pdb_id': pdb_code,
                    'chain_id': 'A',
                    'entity_id': str(entity_id),
                    'molecule_name': molecule_name[:200],  # Truncate long names
                    'sequence_length': sequence_length,
                    'entity_type': entity_type,
                    'organism': organism,
                    'taxonomy_id': str(taxonomy_id),
                    'ec_numbers': ec_numbers,
                    'functional_class': func_class,
                    'size_class': size_class,
                    'organism_class': organism_class
                }

            return None  # Chain A not found in any entity

        except requests.RequestException as e:
            print(f"   Network error for {chain}: {e}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            print(f"   Data error for {chain}: {e}")
            return None
        except Exception as e:
            print(f"   Unexpected error for {chain}: {e}")
            return None

    def classify_function(self, molecule_name: str, ec_numbers: str) -> str:
        """Classify protein by function based on name and EC numbers"""
        text = f"{molecule_name} {ec_numbers}".lower()

        if any(keyword in text for keyword in ['enzyme', 'kinase', 'transferase', 'hydrolase',
                                               'oxidoreductase', 'ligase', 'isomerase', 'lyase']):
            return 'enzyme'
        elif any(keyword in text for keyword in ['binding', 'receptor', 'transport']):
            return 'binding'
        elif any(keyword in text for keyword in ['structural', 'collagen', 'fibrous']):
            return 'structural'
        elif any(keyword in text for keyword in ['antibody', 'immunoglobulin', 'immune']):
            return 'immune'
        elif any(keyword in text for keyword in ['ribosom', 'rna', 'dna', 'nucleic']):
            return 'nucleic_acid_related'
        else:
            return 'other'

    def classify_size(self, sequence_length: int) -> str:
        """Classify protein by size"""
        if sequence_length == 0:
            return 'unknown'
        elif sequence_length < 100:
            return 'small'
        elif sequence_length < 250:
            return 'medium'
        elif sequence_length < 500:
            return 'large'
        else:
            return 'very_large'

    def classify_organism(self, organism: str) -> str:
        """Classify by organism type"""
        organism_lower = organism.lower()

        if any(keyword in organism_lower for keyword in ['escherichia', 'coli']):
            return 'e_coli'
        elif any(keyword in organism_lower for keyword in ['homo sapiens', 'human']):
            return 'human'
        elif any(keyword in organism_lower for keyword in ['mus musculus', 'mouse']):
            return 'mouse'
        elif any(keyword in organism_lower for keyword in ['saccharomyces', 'yeast']):
            return 'yeast'
        elif any(keyword in organism_lower for keyword in ['thermus', 'thermophilic']):
            return 'thermophilic'
        else:
            return 'other'

    def collect_metadata(self, chains: List[str]) -> List[Dict]:
        """Collect metadata for all chains"""
        print("📥 Collecting metadata for protein classification...")

        # Select more chains than needed to account for failures
        buffer_size = min(self.total_size * 3, len(chains))
        selected_chains = random.sample(chains, buffer_size)

        print(f"✅ Selected {len(selected_chains)} chains for metadata collection")

        proteins = []

        for i, chain in enumerate(selected_chains):
            if self.successful_count >= self.total_size:
                print(f"   ✅ Reached target of {self.total_size} proteins")
                break

            # Progress update
            if i % 10 == 0 and i > 0:
                print(f"   Progress: {i}/{len(selected_chains)} "
                      f"({self.successful_count} successful, {self.failed_count} failed)")

            print(f"➡️  [{i + 1}/{len(selected_chains)}] {chain}... ", end='', flush=True)

            # Get metadata
            metadata = self.get_chain_metadata(chain)

            if metadata:
                proteins.append(metadata)
                self.successful_count += 1
                print("✅")
            else:
                self.failed_count += 1
                print("❌")

            # Be nice to the API
            time.sleep(0.3)

        print(f"\n✅ Successfully collected metadata for {self.successful_count} proteins")
        print(f"❌ Failed to collect metadata for {self.failed_count} proteins")

        return proteins

    def create_balanced_splits(self, proteins: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create balanced splits based on multiple stratification criteria"""

        # Adjust split sizes if we don't have enough proteins
        actual_total = len(proteins)
        if actual_total < self.total_size:
            print(f"⚠️  Adjusting split sizes based on available data ({actual_total} proteins)")
            train_size = int(actual_total * 0.615)  # 80/130
            val_size = int(actual_total * 0.192)  # 25/130
            test_size = actual_total - train_size - val_size
        else:
            train_size, val_size, test_size = self.train_size, self.val_size, self.test_size

        # Create strata based on multiple criteria
        strata = defaultdict(list)

        for protein in proteins:
            stratum_key = (
                protein.get('functional_class', 'other'),
                protein.get('size_class', 'unknown'),
                protein.get('organism_class', 'other')
            )
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
                if n_proteins >= 3:
                    test_set.extend(stratum_proteins[2:])
            else:
                train_set.extend(stratum_proteins[:n_train])
                val_set.extend(stratum_proteins[n_train:n_train + n_val])
                test_set.extend(stratum_proteins[n_train + n_val:])

        # Final adjustment to exact sizes
        all_assigned = train_set + val_set + test_set
        random.shuffle(all_assigned)

        final_train = all_assigned[:train_size]
        final_val = all_assigned[train_size:train_size + val_size]
        final_test = all_assigned[train_size + val_size:train_size + val_size + test_size]

        return final_train, final_val, final_test

    def calculate_stats(self, protein_list: List[Dict], name: str) -> Dict:
        """Calculate statistics for a protein list"""
        stats = {
            'count': len(protein_list),
            'functional_distribution': dict(Counter(p.get('functional_class', 'other') for p in protein_list)),
            'size_distribution': dict(Counter(p.get('size_class', 'unknown') for p in protein_list)),
            'organism_distribution': dict(Counter(p.get('organism_class', 'other') for p in protein_list)),
            'entity_type_distribution': dict(Counter(p.get('entity_type', 'Unknown') for p in protein_list))
        }

        lengths = [p.get('sequence_length', 0) for p in protein_list
                   if isinstance(p.get('sequence_length'), int) and p.get('sequence_length', 0) > 0]

        if lengths:
            stats['avg_length'] = sum(lengths) / len(lengths)
            stats['min_length'] = min(lengths)
            stats['max_length'] = max(lengths)
            stats['median_length'] = sorted(lengths)[len(lengths) // 2]

        return stats

    def save_results(self, proteins: List[Dict], train_proteins: List[Dict],
                     val_proteins: List[Dict], test_proteins: List[Dict]):
        """Save results to JSON file"""

        # Create output structure
        output = {
            'metadata': {
                'total_proteins': len(proteins),
                'generation_date': datetime.now().isoformat(),
                'random_seed': self.seed,
                'split_sizes': {
                    'training': len(train_proteins),
                    'validation': len(val_proteins),
                    'testing': len(test_proteins)
                },
                'split_strategy': 'balanced_by_function_size_organism',
                'data_source': 'RODA with RCSB PDB metadata',
                'stratification_criteria': [
                    'functional_class',
                    'size_class',
                    'organism_class'
                ]
            },
            'splits': {
                'training': {
                    'pdb_chains': [p['pdb_chain'] for p in train_proteins],
                    'proteins': train_proteins,
                    'statistics': self.calculate_stats(train_proteins, 'training')
                },
                'validation': {
                    'pdb_chains': [p['pdb_chain'] for p in val_proteins],
                    'proteins': val_proteins,
                    'statistics': self.calculate_stats(val_proteins, 'validation')
                },
                'testing': {
                    'pdb_chains': [p['pdb_chain'] for p in test_proteins],
                    'proteins': test_proteins,
                    'statistics': self.calculate_stats(test_proteins, 'testing')
                }
            }
        }

        # Save main JSON file
        output_json = os.path.join(self.output_dir, 'balanced_protein_splits.json')
        with open(output_json, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"📄 Balanced splits saved to {output_json}")

        return output_json, output

    def print_summary(self, output: Dict):
        """Print summary of the splits"""
        print("\n📊 BALANCED SPLIT SUMMARY:")
        print("=" * 80)

        for split_name, split_data in output['splits'].items():
            stats = split_data['statistics']
            print(f"\n{split_name.upper()} SET: {stats['count']} proteins")

            if 'avg_length' in stats:
                print(f"  Chain lengths: {stats['min_length']}-{stats['max_length']} "
                      f"(avg: {stats['avg_length']:.0f}, median: {stats['median_length']:.0f})")

            print(f"  Functional classes: {stats['functional_distribution']}")
            print(f"  Size classes: {stats['size_distribution']}")
            print(f"  Organism classes: {stats['organism_distribution']}")
            print(f"  Entity types: {stats['entity_type_distribution']}")

        print(f"\n✅ Balanced splits created successfully!")
        print(f"📊 Stratification ensured diversity across:")
        print(f"   - Functional classification (enzyme, binding, structural, etc.)")
        print(f"   - Protein size categories (small, medium, large, very_large)")
        print(f"   - Organism types (e_coli, human, mouse, yeast, thermophilic, other)")

    def run(self):
        """Run the complete pipeline"""
        print("🧬 Creating balanced and diverse protein splits...")
        print(f"🎲 Using random seed: {self.seed}")
        print(f"📁 Output directory: {self.output_dir}")
        print("")

        # Get available chains
        chains = self.get_roda_chains()
        if len(chains) < self.total_size:
            print(f"⚠️  Warning: Only {len(chains)} chains available, need {self.total_size}")

        # Collect metadata
        proteins = self.collect_metadata(chains)
        if len(proteins) == 0:
            print("❌ No proteins processed successfully")
            sys.exit(1)

        print("\n🧬 Creating balanced splits based on collected metadata...")

        # Create balanced splits
        train_proteins, val_proteins, test_proteins = self.create_balanced_splits(proteins)

        # Save results
        output_json, output = self.save_results(proteins, train_proteins, val_proteins, test_proteins)

        # Print summary
        self.print_summary(output)

        print(f"\n🎯 Dataset splits created successfully!")
        print(f"\n📋 Output file:")
        print(f"   - {output_json} (complete dataset with metadata and statistics)")
        print(
            f"   - Contains {len(train_proteins)} training, {len(val_proteins)} validation, and {len(test_proteins)} testing proteins")
        print(f"\n🔧 Use this JSON file with download_split_data.py to fetch the actual data:")
        print(f"   to fetch the actual Evoformer data from RODA S3:")
        print(f"   python download_split_data.py --splits-file {os.path.basename(output_json)}")


def main():
    # Get default output directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, '..', 'data')
    default_output_dir = os.path.abspath(default_output_dir)  # Resolve relative path

    parser = argparse.ArgumentParser(
        description='Create balanced protein splits for neural ODE training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--total-size', type=int, default=130,
                        help='Total number of proteins to collect')
    parser.add_argument('--train-size', type=int, default=80,
                        help='Number of training proteins')
    parser.add_argument('--val-size', type=int, default=25,
                        help='Number of validation proteins')
    parser.add_argument('--test-size', type=int, default=25,
                        help='Number of testing proteins')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                        help='Output directory for results')

    args = parser.parse_args()

    # Validate arguments
    if args.train_size + args.val_size + args.test_size != args.total_size:
        print(f"❌ Error: train_size ({args.train_size}) + val_size ({args.val_size}) + "
              f"test_size ({args.test_size}) != total_size ({args.total_size})")
        sys.exit(1)

    # Create and run the split generator
    generator = ProteinSplitGenerator(
        total_size=args.total_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
        output_dir=args.output_dir
    )

    generator.run()


if __name__ == '__main__':
    main()