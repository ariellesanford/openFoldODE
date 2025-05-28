#!/usr/bin/env python3
"""
Download RODA alignment data and FASTA files for balanced protein splits

This script downloads:
1. RODA alignment data from OpenFold S3 bucket
2. FASTA files from RCSB PDB

Based on the balanced splits created by create_balanced_protein_splits.py

Usage:
    python download_split_data.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import requests


class SplitDataDownloader:
    def __init__(self, output_dir: str = ".", max_retries: int = 3, delay: float = 0.5):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.delay = delay

        # Create base directories
        self.splits = ['training', 'validation', 'testing']
        for split in self.splits:
            split_dir = self.output_dir / split
            (split_dir / 'alignments').mkdir(parents=True, exist_ok=True)
            (split_dir / 'fasta_data').mkdir(parents=True, exist_ok=True)

        # Track download statistics
        self.stats = {
            'alignments': {'successful': 0, 'failed': 0, 'skipped': 0},
            'fasta': {'successful': 0, 'failed': 0, 'skipped': 0}
        }

        # Session for HTTP requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

    def load_splits_from_json(self, json_file: str) -> Dict[str, List[str]]:
        """Load protein splits from JSON file"""
        print(f"📄 Loading splits from {json_file}")

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            splits = {}
            for split_name in ['training', 'validation', 'testing']:
                if split_name in data['splits']:
                    splits[split_name] = data['splits'][split_name]['pdb_chains']
                    print(f"   {split_name}: {len(splits[split_name])} chains")
                else:
                    splits[split_name] = []
                    print(f"   {split_name}: 0 chains (not found)")

            return splits

        except FileNotFoundError:
            print(f"❌ JSON file not found: {json_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON file: {e}")
            sys.exit(1)

    def download_roda_alignments(self, chain: str, split: str) -> bool:
        """Download RODA alignment data for a chain"""
        alignment_dir = self.output_dir / split / 'alignments' / chain

        # Check if already exists
        if alignment_dir.exists() and any(alignment_dir.iterdir()):
            print(f"   ✅ Alignments already exist for {chain}")
            self.stats['alignments']['skipped'] += 1
            return True

        try:
            # Use AWS CLI to download alignment data
            s3_source = f"s3://openfold/pdb/{chain}/"

            # Create the directory
            alignment_dir.mkdir(parents=True, exist_ok=True)

            # Download with AWS CLI
            cmd = [
                'aws', 's3', 'cp', s3_source, str(alignment_dir),
                '--recursive', '--no-sign-request'
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode == 0:
                # Check if we actually got files
                if any(alignment_dir.iterdir()):
                    print(f"   ✅ Downloaded alignments for {chain}")
                    self.stats['alignments']['successful'] += 1
                    return True
                else:
                    print(f"   ❌ No alignment files found for {chain}")
                    alignment_dir.rmdir()  # Remove empty directory
                    self.stats['alignments']['failed'] += 1
                    return False
            else:
                print(f"   ❌ AWS CLI failed for {chain}: {result.stderr.strip()}")
                if alignment_dir.exists():
                    alignment_dir.rmdir()  # Remove empty directory
                self.stats['alignments']['failed'] += 1
                return False

        except subprocess.TimeoutExpired:
            print(f"   ❌ Timeout downloading alignments for {chain}")
            if alignment_dir.exists():
                alignment_dir.rmdir()
            self.stats['alignments']['failed'] += 1
            return False
        except Exception as e:
            print(f"   ❌ Error downloading alignments for {chain}: {e}")
            if alignment_dir.exists():
                alignment_dir.rmdir()
            self.stats['alignments']['failed'] += 1
            return False

    def download_fasta(self, chain: str, split: str) -> bool:
        """Download FASTA data for a chain"""
        pdb_code = chain.split('_')[0]
        fasta_file = self.output_dir / split / 'fasta_data' / f"{chain}.fasta"

        # Check if already exists
        if fasta_file.exists():
            print(f"   ✅ FASTA already exists for {chain}")
            self.stats['fasta']['skipped'] += 1
            return True

        # Try downloading FASTA with retries
        for attempt in range(self.max_retries):
            try:
                # Use the RCSB PDB FASTA endpoint
                fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_code.upper()}"

                response = self.session.get(fasta_url, timeout=10)

                if response.status_code == 200 and response.text.strip():
                    # Parse FASTA content to extract only chain A
                    fasta_content = response.text

                    # Look for chain A specifically
                    lines = fasta_content.split('\n')
                    chain_a_content = []
                    in_chain_a = False

                    for line in lines:
                        if line.startswith('>'):
                            # Parse RCSB PDB header format: >3EPS_1|Chains A, B|Isocitrate dehydrogenase...|Escherichia coli...
                            # Check if this entry contains chain A
                            if '|' in line:
                                parts = line.split('|')
                                if len(parts) >= 2:
                                    chains_part = parts[1]  # "Chains A, B" or "Chain A"
                                    # Extract chain letters from the chains part
                                    if 'Chain A' in chains_part or 'Chains A' in chains_part or ', A' in chains_part or chains_part.endswith(
                                            ' A'):
                                        in_chain_a = True
                                        # Simplify header to match expected format
                                        chain_a_content.append(f">{chain}")
                                        continue

                            # Fallback: check for other patterns
                            if 'Chain A' in line or f'_{pdb_code.upper()}_A' in line:
                                in_chain_a = True
                                # Simplify header to match expected format
                                chain_a_content.append(f">{chain}")
                            else:
                                in_chain_a = False
                        elif in_chain_a and line.strip():
                            chain_a_content.append(line.strip())

                    if chain_a_content:
                        # Save the FASTA file
                        fasta_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(fasta_file, 'w') as f:
                            f.write('\n'.join(chain_a_content))

                        print(f"   ✅ Downloaded FASTA for {chain}")
                        self.stats['fasta']['successful'] += 1
                        return True
                    else:
                        print(f"   ❌ Chain A not found in FASTA for {chain}")
                        self.stats['fasta']['failed'] += 1
                        return False

                elif response.status_code == 404:
                    print(f"   ❌ FASTA not found for {chain} (404)")
                    self.stats['fasta']['failed'] += 1
                    return False
                else:
                    print(f"   ⚠️  HTTP {response.status_code} for {chain}, attempt {attempt + 1}")

            except requests.RequestException as e:
                print(f"   ⚠️  Network error for {chain}, attempt {attempt + 1}: {e}")

            # Wait before retry
            if attempt < self.max_retries - 1:
                time.sleep(self.delay * (attempt + 1))

        print(f"   ❌ Failed to download FASTA for {chain} after {self.max_retries} attempts")
        self.stats['fasta']['failed'] += 1
        return False

    def process_split(self, split_name: str, chains: List[str]):
        """Process all chains in a split"""
        print(f"\n🔄 Processing {split_name} split ({len(chains)} chains)")
        print("=" * 60)

        # Track progress
        successful_alignments = 0
        successful_fasta = 0

        for i, chain in enumerate(chains):
            print(f"\n[{i + 1}/{len(chains)}] Processing {chain}:")

            # Download alignments
            print("  📦 Downloading alignments...")
            if self.download_roda_alignments(chain, split_name):
                successful_alignments += 1

            # Small delay between alignment and FASTA download
            time.sleep(0.2)

            # Download FASTA
            print("  🧬 Downloading FASTA...")
            if self.download_fasta(chain, split_name):
                successful_fasta += 1

            # Progress update every 10 chains
            if (i + 1) % 10 == 0:
                print(f"\n   📊 Progress: {i + 1}/{len(chains)} chains processed")
                print(f"      Alignments: {successful_alignments}/{i + 1} successful")
                print(f"      FASTA: {successful_fasta}/{i + 1} successful")

            # Be nice to the servers
            time.sleep(self.delay)

        print(f"\n✅ {split_name} split completed:")
        print(f"   Alignments: {successful_alignments}/{len(chains)} successful")
        print(f"   FASTA: {successful_fasta}/{len(chains)} successful")

    def print_final_stats(self):
        """Print final download statistics"""
        print("\n" + "=" * 60)
        print("📊 FINAL DOWNLOAD STATISTICS")
        print("=" * 60)

        for data_type in ['alignments', 'fasta']:
            stats = self.stats[data_type]
            total = stats['successful'] + stats['failed'] + stats['skipped']
            print(f"\n{data_type.upper()}:")
            print(f"  ✅ Successful: {stats['successful']}")
            print(f"  ⏭️  Skipped (already exists): {stats['skipped']}")
            print(f"  ❌ Failed: {stats['failed']}")
            print(f"  📈 Success rate: {stats['successful'] / max(total, 1) * 100:.1f}%")

        print(f"\n📁 Output directory structure:")
        for split in self.splits:
            split_dir = self.output_dir / split
            if split_dir.exists():
                alignment_count = len(list((split_dir / 'alignments').iterdir())) if (
                            split_dir / 'alignments').exists() else 0
                fasta_count = len(list((split_dir / 'fasta_data').glob('*.fasta'))) if (
                            split_dir / 'fasta_data').exists() else 0
                print(f"  {split}/")
                print(f"    alignments/ ({alignment_count} folders)")
                print(f"    fasta_data/ ({fasta_count} files)")

    def run(self, json_file: str):
        """Run the download process using JSON splits file"""
        splits = self.load_splits_from_json(json_file)

        # Process each split
        for split_name in ['training', 'validation', 'testing']:
            if splits[split_name]:
                self.process_split(split_name, splits[split_name])

        self.print_final_stats()


def main():
    # Hardcoded paths - modify these as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output_dir = os.path.join(script_dir, '..', 'data')
    default_output_dir = os.path.abspath(default_output_dir)

    # JSON file path - modify this to point to your splits file
    json_file = os.path.join(default_output_dir, 'balanced_protein_splits.json')

    parser = argparse.ArgumentParser(
        description='Download RODA alignments and FASTA files for protein splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments for customization
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                        help='Output directory (will create training/, validation/, testing/ subdirs)')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retry attempts for failed downloads')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between downloads (seconds)')

    # Parse arguments
    args = parser.parse_args()

    # Check if JSON file exists
    if not os.path.exists(json_file):
        print(f"❌ Error: JSON file not found: {json_file}")
        print("Please run create_balanced_protein_splits.py first, or modify the json_file path in this script.")
        sys.exit(1)

    # Check dependencies
    try:
        subprocess.run(['aws', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: AWS CLI not found or not working")
        print("Please install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html")
        sys.exit(1)

    print("🚀 RODA and FASTA Data Downloader")
    print(f"📄 Using splits from JSON file: {json_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Max retries: {args.max_retries}")
    print(f"⏱️  Delay between downloads: {args.delay}s")

    # Create downloader
    downloader = SplitDataDownloader(
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        delay=args.delay
    )

    # Run the download process
    downloader.run(json_file)

    print("\n🎯 Download process completed!")
    print("\n💡 Next steps:")
    print("   1. Use the alignment data for training your neural ODE")
    print("   2. Use the FASTA files for sequence analysis")
    print("   3. Check the statistics above for any failed downloads")


if __name__ == '__main__':
    main()