#!/usr/bin/env python3
"""
Expand existing protein splits by adding more proteins while maintaining balance
"""

import argparse
import json
import os
import random
import subprocess
import sys
from collections import defaultdict, Counter
from datetime import datetime
from typing import Dict, List, Set

import requests

# Import the original classification logic
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from balanced_protein_splits import ProteinSplitGenerator


class ProteinSplitExpander(ProteinSplitGenerator):
    def __init__(self, input_splits_dir: str, output_splits_dir: str,
                 additional_train: int, additional_val: int, additional_test: int, seed: int = 42):
        # Initialize parent class (but don't run its full pipeline)
        super().__init__(total_size=0, seed=seed, output_dir=output_splits_dir)

        self.input_splits_dir = input_splits_dir
        self.output_splits_dir = output_splits_dir
        self.additional_train = additional_train
        self.additional_val = additional_val
        self.additional_test = additional_test

    def load_existing_splits(self) -> tuple[List[str], List[str], List[str]]:
        """Load existing protein splits from text files"""
        splits = {}

        for split_name in ['training', 'validation', 'testing']:
            file_path = os.path.join(self.input_splits_dir, f"{split_name}_chains.txt")
            with open(file_path, 'r') as f:
                proteins = [line.strip() for line in f if line.strip()]
            splits[split_name] = proteins
            print(f"📥 Loaded {len(proteins)} existing {split_name} proteins")

        return splits['training'], splits['validation'], splits['testing']

    def get_available_roda_chains(self) -> List[str]:
        """Get list of available RODA chains using parent class method"""
        return self.get_roda_chains()

    def get_chain_metadata(self, chain: str) -> Dict:
        """Get metadata for a chain using parent class method"""
        return super().get_chain_metadata(chain)

    def collect_new_proteins(self, existing_proteins: Set[str], target_count: int) -> List[Dict]:
        """Collect metadata for new proteins not in existing sets"""
        print(f"🔍 Collecting {target_count} new proteins...")

        available_chains = self.get_available_roda_chains()
        new_chains = [chain for chain in available_chains if chain not in existing_proteins]

        if len(new_chains) < target_count:
            print(f"⚠️  Only {len(new_chains)} new chains available, need {target_count}")
            target_count = len(new_chains)

        # Sample more than needed to account for failures
        sample_size = min(target_count * 3, len(new_chains))
        selected_chains = random.sample(new_chains, sample_size)

        proteins = []
        for i, chain in enumerate(selected_chains):
            if len(proteins) >= target_count:
                break

            if i % 20 == 0:
                print(f"   Progress: {i}/{len(selected_chains)} ({len(proteins)} collected)")

            metadata = self.get_chain_metadata(chain)
            if metadata:
                proteins.append(metadata)

        print(f"✅ Collected {len(proteins)} new proteins")
        return proteins

    def distribute_proteins_balanced(self, proteins: List[Dict], train_needed: int, val_needed: int,
                                     test_needed: int) -> tuple:
        """Distribute proteins maintaining balance across splits"""

        # Group by characteristics for balanced distribution
        strata = defaultdict(list)
        for protein in proteins:
            stratum_key = (
                protein.get('functional_class', 'other'),
                protein.get('size_class', 'unknown'),
                protein.get('organism_class', 'other')
            )
            strata[stratum_key].append(protein)

        # Shuffle within strata
        for stratum_proteins in strata.values():
            random.shuffle(stratum_proteins)

        # Distribute proportionally
        total_needed = train_needed + val_needed + test_needed
        train_proteins = []
        val_proteins = []
        test_proteins = []

        for stratum_proteins in strata.values():
            n_proteins = len(stratum_proteins)
            if n_proteins == 0:
                continue

            # Calculate proportional allocation
            n_train = int(n_proteins * train_needed / total_needed)
            n_val = int(n_proteins * val_needed / total_needed)
            n_test = n_proteins - n_train - n_val

            # Allocate proteins
            train_proteins.extend(stratum_proteins[:n_train])
            val_proteins.extend(stratum_proteins[n_train:n_train + n_val])
            test_proteins.extend(stratum_proteins[n_train + n_val:n_train + n_val + n_test])

        # Adjust to exact counts by randomly redistributing excess
        all_assigned = train_proteins + val_proteins + test_proteins
        random.shuffle(all_assigned)

        final_train = all_assigned[:train_needed]
        final_val = all_assigned[train_needed:train_needed + val_needed]
        final_test = all_assigned[train_needed + val_needed:train_needed + val_needed + test_needed]

        return final_train, final_val, final_test

    def save_expanded_splits(self, train_proteins: List, val_proteins: List, test_proteins: List,
                             all_proteins: List[Dict]):
        """Save expanded splits using parent class methods"""
        os.makedirs(self.output_splits_dir, exist_ok=True)

        # Convert protein chain strings back to protein dicts for the ones we need
        train_protein_dicts = []
        val_protein_dicts = []
        test_protein_dicts = []

        # Create lookup for protein metadata
        protein_lookup = {p['pdb_chain']: p for p in all_proteins if isinstance(p, dict)}

        for protein_id in train_proteins:
            if isinstance(protein_id, str) and protein_id in protein_lookup:
                train_protein_dicts.append(protein_lookup[protein_id])
            elif isinstance(protein_id, dict):
                train_protein_dicts.append(protein_id)

        for protein_id in val_proteins:
            if isinstance(protein_id, str) and protein_id in protein_lookup:
                val_protein_dicts.append(protein_lookup[protein_id])
            elif isinstance(protein_id, dict):
                val_protein_dicts.append(protein_id)

        for protein_id in test_proteins:
            if isinstance(protein_id, str) and protein_id in protein_lookup:
                test_protein_dicts.append(protein_lookup[protein_id])
            elif isinstance(protein_id, dict):
                test_protein_dicts.append(protein_id)

        # Use parent class method to save results and create balanced_protein_splits.json
        output_json, output = self.save_results(
            all_proteins, train_protein_dicts, val_protein_dicts, test_protein_dicts
        )

        # Also save text files for compatibility
        splits = {
            'training': train_proteins,
            'validation': val_proteins,
            'testing': test_proteins
        }

        for split_name, proteins in splits.items():
            file_path = os.path.join(self.output_splits_dir, f"{split_name}_chains.txt")
            with open(file_path, 'w') as f:
                for protein in proteins:
                    if isinstance(protein, dict):
                        f.write(f"{protein['pdb_chain']}\n")
                    else:
                        f.write(f"{protein}\n")

        # Add expansion metadata to the output
        if isinstance(output, dict) and 'metadata' in output:
            output['metadata']['expansion_info'] = {
                'source_splits': self.input_splits_dir,
                'expansion_date': datetime.now().isoformat(),
                'random_seed': self.seed,
                'proteins_added': {
                    'training': self.additional_train,
                    'validation': self.additional_val,
                    'testing': self.additional_test
                }
            }

            # Re-save the JSON with expansion info
            with open(output_json, 'w') as f:
                json.dump(output, f, indent=2)

        return output_json, output

    def run(self):
        """Run the expansion process"""
        print(f"🚀 Expanding protein splits from {self.input_splits_dir}")
        print(f"📈 Adding: {self.additional_train} train, {self.additional_val} val, {self.additional_test} test")
        print(f"💾 Output: {self.output_splits_dir}")

        # Load existing splits
        existing_train, existing_val, existing_test = self.load_existing_splits()
        all_existing = set(existing_train + existing_val + existing_test)

        # Calculate total proteins needed
        total_needed = self.additional_train + self.additional_val + self.additional_test

        # Collect new proteins
        new_proteins = self.collect_new_proteins(all_existing, total_needed)

        if len(new_proteins) < total_needed:
            print(f"⚠️  Only found {len(new_proteins)} new proteins, adjusting targets proportionally")
            scale_factor = len(new_proteins) / total_needed
            self.additional_train = int(self.additional_train * scale_factor)
            self.additional_val = int(self.additional_val * scale_factor)
            self.additional_test = len(new_proteins) - self.additional_train - self.additional_val

        # Distribute new proteins
        new_train, new_val, new_test = self.distribute_proteins_balanced(
            new_proteins, self.additional_train, self.additional_val, self.additional_test
        )

        # Combine with existing
        final_train = existing_train + [p['pdb_chain'] for p in new_train]
        final_val = existing_val + [p['pdb_chain'] for p in new_val]
        final_test = existing_test + [p['pdb_chain'] for p in new_test]

        # Create combined protein list for metadata
        all_proteins_with_metadata = new_proteins.copy()

        # Save results using parent class methods
        output_json, output = self.save_expanded_splits(
            final_train, final_val, final_test, all_proteins_with_metadata
        )

        # Print results using parent class method
        self.print_summary(output)

        print(f"\n✅ Expansion complete!")
        print(f"📊 Final counts:")
        print(f"   Training: {len(final_train)} (was {len(existing_train)}, +{len(new_train)})")
        print(f"   Validation: {len(final_val)} (was {len(existing_val)}, +{len(new_val)})")
        print(f"   Testing: {len(final_test)} (was {len(existing_test)}, +{len(new_test)})")
        print(f"📄 Complete dataset saved to: {output_json}")


def main():
    parser = argparse.ArgumentParser(description='Expand existing protein splits')
    parser.add_argument('--input-splits-dir', type=str, required=True,
                        help='Directory containing existing splits (e.g., data_splits/full)')
    parser.add_argument('--output-splits-dir', type=str, required=True,
                        help='Directory for expanded splits (e.g., data_splits/jumbo)')
    parser.add_argument('--additional-train', type=int, default=420,
                        help='Number of training proteins to add')
    parser.add_argument('--additional-val', type=int, default=25,
                        help='Number of validation proteins to add')
    parser.add_argument('--additional-test', type=int, default=25,
                        help='Number of testing proteins to add')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    expander = ProteinSplitExpander(
        args.input_splits_dir,
        args.output_splits_dir,
        args.additional_train,
        args.additional_val,
        args.additional_test,
        args.seed
    )

    expander.run()


if __name__ == '__main__':
    main()