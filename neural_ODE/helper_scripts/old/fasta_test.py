#!/usr/bin/env python3
"""
Test script to download specific FASTA files for testing

Downloads FASTA files for: 7d60_A, 3foa_A, 3eps_A

Usage: python test_fasta_download.py
"""

import os
import requests
import sys
from pathlib import Path


def download_fasta(chain: str, output_dir: str = ".") -> bool:
    """Download FASTA data for a specific chain"""
    pdb_code = chain.split('_')[0]
    output_path = Path(output_dir) / f"{chain}.fasta"

    print(f"🧬 Downloading FASTA for {chain}...")

    # Check if already exists
    if output_path.exists():
        print(f"   ✅ FASTA already exists: {output_path}")
        return True

    try:
        # Use the RCSB PDB FASTA endpoint
        fasta_url = f"https://www.rcsb.org/fasta/entry/{pdb_code.upper()}"

        # Create session with headers
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
        })

        response = session.get(fasta_url, timeout=10)

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
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'w') as f:
                    f.write('\n'.join(chain_a_content))

                print(f"   ✅ Downloaded FASTA: {output_path}")

                # Show preview of content
                with open(output_path, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    print(f"   📄 Header: {lines[0] if lines else 'Empty'}")
                    sequence = ''.join(lines[1:]) if len(lines) > 1 else ''
                    print(f"   📏 Sequence length: {len(sequence)} amino acids")
                    if sequence:
                        print(f"   🔤 First 50 chars: {sequence[:50]}...")

                return True
            else:
                print(f"   ❌ Chain A not found in FASTA for {chain}")

                # Debug: show what headers we found
                print(f"   🔍 Debug - Headers found in FASTA:")
                debug_lines = fasta_content.split('\n')
                for line in debug_lines:
                    if line.startswith('>'):
                        print(f"      {line}")

                return False

        elif response.status_code == 404:
            print(f"   ❌ FASTA not found for {chain} (404 - PDB entry doesn't exist)")
            return False
        else:
            print(f"   ❌ HTTP {response.status_code} for {chain}")
            return False

    except requests.RequestException as e:
        print(f"   ❌ Network error for {chain}: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error for {chain}: {e}")
        return False


def main():
    # Test chains
    test_chains = ["7d60_A", "3foa_A", "3eps_A"]

    # Output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "test_fastas")

    print("🧪 FASTA Download Test")
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Testing chains: {', '.join(test_chains)}")
    print("=" * 50)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Download each FASTA
    results = {}
    for chain in test_chains:
        print(f"\n[{test_chains.index(chain) + 1}/{len(test_chains)}] {chain}")
        results[chain] = download_fasta(chain, output_dir)

    # Summary
    print("\n" + "=" * 50)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 50)

    successful = sum(1 for success in results.values() if success)
    failed = len(results) - successful

    print(f"✅ Successful downloads: {successful}/{len(test_chains)}")
    print(f"❌ Failed downloads: {failed}/{len(test_chains)}")

    print(f"\n📋 Detailed results:")
    for chain, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        file_path = os.path.join(output_dir, f"{chain}.fasta")
        print(f"   {chain}: {status}")
        if success and os.path.exists(file_path):
            print(f"      File: {file_path}")

    print(f"\n📁 All files saved to: {output_dir}")

    if successful == len(test_chains):
        print("\n🎉 All test downloads completed successfully!")
    else:
        print(f"\n⚠️  {failed} download(s) failed. Check the error messages above.")

    # Show directory contents
    output_path = Path(output_dir)
    if output_path.exists():
        fasta_files = list(output_path.glob("*.fasta"))
        if fasta_files:
            print(f"\n📂 Downloaded files:")
            for fasta_file in fasta_files:
                file_size = fasta_file.stat().st_size
                print(f"   {fasta_file.name} ({file_size} bytes)")


if __name__ == '__main__':
    main()