import pickle
import matplotlib
#matplotlib.use("Agg")  # Use non-interactive backend for script use
import matplotlib.pyplot as plt
import os
import sys

# === CONFIG: path to your output_dict.pkl ===
protein_id = ("5fr6")
PKL_PATH1 = f"/home/visitor/PycharmProjects/openFold/checkpointing/monomers/predictions/predictions/{protein_id}_A_model_1_ptm_output_dict.pkl"
PKL_PATH2 = f"/home/visitor/PycharmProjects/openFold/checkpointing/monomers/predictions_noRecycles/predictions/{protein_id}_A_model_1_ptm_output_dict.pkl"
PKL_PATH3 = f"/home/visitor/PycharmProjects/openFold/checkpointing/evoformer_inits/predictions/{protein_id}_A_model_1_ptm_output_dict.pkl"
for PKL_PATH in [PKL_PATH3]:
    OUTPUT_PDF = os.path.splitext(PKL_PATH)[0] + "_plddt_plot.pdf"
    # === Load prediction dict ===
    with open(PKL_PATH, "rb") as f:
        out = pickle.load(f)

    # === Extract pLDDT ===
    plddt = out.get("plddt", out.get("plddts", None))
    if plddt is None:
        print("⚠️ pLDDT not found in output_dict.pkl.")
        sys.exit(1)

    # === Extract pTM score ===
    ptm = out.get("ptm_score")
    if isinstance(ptm, dict):
        ptm = ptm.get("mean", None)

    # === Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(plddt, color='blue')
    plt.title("Predicted Local Distance Difference Test (pLDDT)", fontsize=14)
    plt.xlabel("Residue Index", fontsize=12)
    plt.ylabel("pLDDT Score", fontsize=12)
    plt.ylim(0, 100)
    plt.grid(True)

    # Add pTM score as annotation in the top-right
    if ptm is not None:
        plt.text(
            0.98, 0.95,
            f"pTM: {ptm:.4f}",
            transform=plt.gca().transAxes,
            ha='right',
            va='top',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="lightgray", ec="black", lw=1)
        )
    else:
        print("⚠️ pTM score not available.")

    # === Save as PDF ===
    plt.tight_layout()
    plt.savefig(OUTPUT_PDF)
    print(f"✅ Plot saved to: {OUTPUT_PDF}")
