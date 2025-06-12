#!/usr/bin/env python3
"""
Download RODA alignment data and FASTA files for balanced protein splits

Modified version that:
- Outputs to /media/visitor/Extreme SSD/data with alignments/ and fasta_data/ subdirectories
- Doesn't split by training/validation/testing directories
- Reads from jumbo splits directory

Usage:
    python download_split_data_modified.py
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
    def __init__(self, output_dir: str = "/media/visitor/Extreme SSD/data", max_retries: int = 3, delay: float = 0.5):
        self.output_dir = Path(output_dir)
        self.max_retries = max_retries
        self.delay = delay

        # Create base directories (no split subdirectories)
        (self.output_dir / 'alignments').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'fasta_data').mkdir(parents=True, exist_ok=True)

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

    def load_splits_from_text_files(self, splits_dir: str) -> Dict[str, List[str]]:
        """Load protein splits from text files"""
        print(f"📄 Loading splits from {splits_dir}")

        splits = {}
        for split_name in ['training', 'validation', 'testing']:
            file_path = os.path.join(splits_dir, f"{split_name}_chains.txt")
            try:
                with open(file_path, 'r') as f:
                    chains = [line.strip() for line in f if line.strip()]
                splits[split_name] = chains
                print(f"   {split_name}: {len(chains)} chains")
            except FileNotFoundError:
                print(f"   {split_name}: 0 chains (file not found)")
                splits[split_name] = []

        return splits

    def download_roda_alignments(self, chain: str) -> bool:
        """Download RODA alignment data for a chain"""
        alignment_dir = self.output_dir / 'alignments' / chain

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

    def download_fasta(self, chain: str) -> bool:
        """Download FASTA data for a chain"""
        pdb_code = chain.split('_')[0]
        fasta_file = self.output_dir / 'fasta_data' / f"{chain}.fasta"

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

    def process_all_chains(self, all_chains: List[str]):
        """Process all chains without split-based organization"""
        print(f"\n🔄 Processing {len(all_chains)} total chains")
        print("=" * 60)

        # Track progress
        successful_alignments = 0
        successful_fasta = 0

        for i, chain in enumerate(all_chains):
            print(f"\n[{i + 1}/{len(all_chains)}] Processing {chain}:")

            # Download alignments
            print("  📦 Downloading alignments...")
            if self.download_roda_alignments(chain):
                successful_alignments += 1

            # Small delay between alignment and FASTA download
            time.sleep(0.2)

            # Download FASTA
            print("  🧬 Downloading FASTA...")
            if self.download_fasta(chain):
                successful_fasta += 1

            # Progress update every 10 chains
            if (i + 1) % 10 == 0:
                print(f"\n   📊 Progress: {i + 1}/{len(all_chains)} chains processed")
                print(f"      Alignments: {successful_alignments}/{i + 1} successful")
                print(f"      FASTA: {successful_fasta}/{i + 1} successful")

            # Be nice to the servers
            time.sleep(self.delay)

        print(f"\n✅ All chains completed:")
        print(f"   Alignments: {successful_alignments}/{len(all_chains)} successful")
        print(f"   FASTA: {successful_fasta}/{len(all_chains)} successful")

    def print_final_stats(self, all_chains: List[str]):
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
        alignment_count = len(list((self.output_dir / 'alignments').iterdir())) if (
                    self.output_dir / 'alignments').exists() else 0
        fasta_count = len(list((self.output_dir / 'fasta_data').glob('*.fasta'))) if (
                    self.output_dir / 'fasta_data').exists() else 0

        print(f"  {self.output_dir}/")
        print(f"    alignments/ ({alignment_count} folders)")
        print(f"    fasta_data/ ({fasta_count} files)")

        print(f"\n📊 Total unique chains processed: {len(all_chains)}")

    def run(self, splits_dir: str):
        """Run the download process using splits directory"""
        splits = self.load_splits_from_text_files(splits_dir)

        # Combine all chains from all splits (remove duplicates)
        all_chains = set()
        for split_name, chains in splits.items():
            all_chains.update(chains)

        all_chains = sorted(list(all_chains))  # Convert to sorted list

        if not all_chains:
            print("❌ No chains found in any split files")
            return

        print(f"\n📊 Split breakdown:")
        for split_name, chains in splits.items():
            print(f"  {split_name}: {len(chains)} chains")
        print(f"  Total unique: {len(all_chains)} chains")

        # Process all chains
        self.process_all_chains(all_chains)

        self.print_final_stats(all_chains)


def main():
    # Hardcoded paths - modify these as needed
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Default paths
    default_output_dir = "/media/visitor/Extreme SSD/data"
    default_splits_dir = os.path.join(script_dir, 'data_splits', 'jumbo')

    parser = argparse.ArgumentParser(
        description='Download RODA alignments and FASTA files for protein splits',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional arguments for customization
    parser.add_argument('--output-dir', type=str, default=default_output_dir,
                        help='Output directory (will create alignments/ and fasta_data/ subdirs)')
    parser.add_argument('--splits-dir', type=str, default=default_splits_dir,
                        help='Directory containing split text files')
    parser.add_argument('--max-retries', type=int, default=3,
                        help='Maximum number of retry attempts for failed downloads')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between downloads (seconds)')

    # Parse arguments
    args = parser.parse_args()

    # Check if splits directory exists
    if not os.path.exists(args.splits_dir):
        print(f"❌ Error: Splits directory not found: {args.splits_dir}")
        print("Available directories:")
        splits_base = os.path.dirname(args.splits_dir)
        if os.path.exists(splits_base):
            for item in os.listdir(splits_base):
                if os.path.isdir(os.path.join(splits_base, item)):
                    print(f"  - {item}")
        sys.exit(1)

    # Check dependencies
    try:
        subprocess.run(['aws', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Error: AWS CLI not found or not working")
        print("Please install AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html")
        sys.exit(1)

    print("🚀 RODA and FASTA Data Downloader (Modified)")
    print(f"📂 Using splits from: {args.splits_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🔄 Max retries: {args.max_retries}")
    print(f"⏱️  Delay between downloads: {args.delay}s")
    print(f"📦 Structure: alignments/ and fasta_data/ (no split subdirectories)")

    # Create downloader
    downloader = SplitDataDownloader(
        output_dir=args.output_dir,
        max_retries=args.max_retries,
        delay=args.delay
    )

    # Run the download process
    downloader.run(args.splits_dir)

    print("\n🎯 Download process completed!")
    print("\n💡 Next steps:")
    print("   1. Use the alignment data for training your neural ODE")
    print("   2. Use the FASTA files for sequence analysis")
    print("   3. Check the statistics above for any failed downloads")
    print(f"\n📁 Data structure:")
    print(f"   {args.output_dir}/alignments/[chain_id]/  # Alignment files")
    print(f"   {args.output_dir}/fasta_data/[chain_id].fasta  # FASTA files")


if __name__ == '__main__':
    main()