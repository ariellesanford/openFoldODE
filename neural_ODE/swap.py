import json
from pathlib import Path

def swap_labels_in_json(filename):
    # Load the original JSON data
    with open(filename, 'r') as f:
        data = json.load(f)

    # Iterate through each protein entry and swap the times
    for protein_id, entry in data.get("proteins", {}).items():
        if "openfold time" in entry and "neural ODE time" in entry:
            # Swap the two values
            entry["openfold time"], entry["neural ODE time"] = entry["neural ODE time"], entry["openfold time"]

    # Write the modified data back to a new file
    output_file = Path(filename).with_name(f"{Path(filename).stem}_swapped.json")
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✅ Swapped labels written to: {output_file}")

if __name__ == "__main__":
    swap_labels_in_json("benchmark_results_20250616_180845_full_ode_with_prelim_final_model_swapped.json")
