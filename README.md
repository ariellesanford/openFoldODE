Protein Folding with Neural Ordinary Differential Equations
==============

openFoldODE implements a Neural Ordinary Differential Equation (Neural ODE) formulation of the Evoformer. Instead of fixed discrete blocks, we use a continuous-depth parameterization that preserves the Evoformer’s attention-based operations while enabling efficient and adaptive computation.

## Getting Started

### Prerequisites
- Linux operating system (required for OpenFold data generation)
- Python with virtual environment support

### Installation
1. Download the GitHub repository
2. Set up a virtual environment
3. Navigate to the `neural_ode` directory

## Data Setup

### Using Sample Data (Quick Testing)
For quick testing, use the provided sample data:
- **Data splits**: `data_splits/mini`
- **Data directory**: `mini_data`

### Generating Your Own Data
If you want to generate custom training and inference data:

1. **Create protein splits**
   ```bash
   python helper_scripts/balanced_protein_splits.py --total-size 3 --train-size 1 --val-size 1 --test-size 1 --output-dir data_splits/{your_splits_dir}
   ```
   *Note: Requires AWS CLI installation*

2. **Download split data**
   ```bash
   python helper_scripts/download_split_data.py --splits-dir data_splits/{your_splits_dir} --output-dir/{your_data_dir}
   ```

3. **Download AlphaFold parameters**
   ```bash
   ../save_intermediates/scripts/download_alphafold_params.sh openfold/resources
   ```

4. **Download PDB70 data**
   ```bash
   bash scripts/download_pdb70_mmcif_only.sh
   ```
   *Note: Change `DOWNLOAD_DIR` to your `{your_data_dir}` destination. Requires 113 GB storage.*

5. **Generate EvoFormer inputs**
   ```bash
   python helper_scripts/generate_evoformer_inputs.py --data-dir {your_data_dir} --splits-dir "data_splits/{your_splits_dir}"
   ```

6. **Generate blocks**
   *Generating all blocks for at least some proteins will allow you to use preliminary training to guide the protein folding evolution to mimic that of the evoformer, which may ultimately lead to better predictions.
   - For all blocks (needed for preliminary training):
     ```bash
     python helper_scripts/generate_all_blocks.py --data-dir {your_data_dir}
     ```
   - For 48th block only (enough for main training):
     ```bash
     python helper_scripts/generate_48th_blocks.py --data-dir {your_data_dir}
     ```

## Usage

### Inference

#### Quick Start with Sample Data
```bash
python run_test.py 20250616_180845_full_ode_with_prelim_final_model.pt --data-dir mini_data --splits-dir data_splits/mini
```

#### General Usage
```bash
python run_test.py {specific_model_name.pt} --data-dir {your_data_dir} --splits-dir data_splits/{your_splits_dir}
```

#### Available Models
All trained models are stored in the `trained_models` directory. The best performing model is:
- **Model**: `20250616_180845_full_ode_with_prelim_final_model.pt`
- **Config/Stats**: `20250616_180845_full_ode_with_prelim.txt` (contains configuration settings, training progress, and performance statistics)

### Training

#### Basic Training (Mini Data)
```bash
python training_runner.py \
  --data-dir mini_data \
  --splits-dir data_splits/mini \
  --max-epochs 5 \
  --no-prelim
```

#### CPU Training
```bash
python training_runner.py \
  --data-dir mini_data \
  --splits-dir data_splits/mini \
  --max-epochs 5 \
  --no-prelim \
  cpu
```
*Note: CPU training runs significantly slower than GPU training*

## Troubleshooting

### Common Issues

**Error: `$'\r': command not found`**
```bash
sed -i 's/\r$//' {script_path}
```
Then rerun the script. This removes Windows line endings that can cause issues on Linux systems.

References
----------
- Gustaf Ahdritz, Nazim Bouatta, Christina Floristean, Sachin Kadyan, Qinghui Xia, 
William Gerecke, Timothy J. O'Donnell, Daniel Berenberg, Ian Fisk, Niccolò 
Zanichelli, Bo Zhang, Arkadiusz Nowaczynski, Bei Wang, Marta M. 
Stepniewska-Dziubinska, Shang Zhang, Adegoke Ojewole, Murat Efe Guney, 
Stella Biderman, Andrew M. Watkins, Stephen Ra, Pablo Ribalta Lorenzo, Lucas Nivon,
Brian Weitzner, Yih-En Andrew Ban, Shiyang Chen, Minjia Zhang, Conglong Li, 
Shuaiwen Leon Song, Yuxiong He, Peter K. Sorger, Emad Mostaque, Zhao Zhang, 
Richard Bonneau, Mohammed AlQuraishi  
  OpenFold: retraining AlphaFold2 yields new insights into its learning mechanisms and capacity for generalization  
  [Nat. Methods 21, 1514–1524 (2024)](https://doi.org/10.1038/s41592-024-02272-z)  
  [Code repository](https://github.com/aqlaboratory/openfold)  
- John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf
Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Žídek, Anna Potapenko,
Alex Bridgland, Clemens Meyer, Simon A. A. Kohl, Andrew J. Ballard, Andrew Cowie,
Bernardino Romera-Paredes, Stanislav Nikolov, Rishub Jain, Jonas Adler, Trevor
Back, Stig Petersen, David Reiman, Ellen Clancy, Michal Zielinski, Martin
Steinegger, Michalina Pacholska, Tamas Berghammer, Sebastian Bodenstein, David
Silver, Oriol Vinyals, Andrew W. Senior, Koray Kavukcuoglu, Pushmeet Kohli, 
Demis Hassabis  
  Highly accurate protein structure prediction with AlphaFold  
  [Nature 596, 583–589 (2021)](https://doi.org/10.1038/s41586-021-03819-2)
- Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David K. Duvenaud   
  Neural ordinary differential equations  
  [Adv. NeurIPS 31 (2018)](https://proceedings.neurips.cc/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf)
- Schrödinger LLC  
  The PyMOL molecular graphics system  
  [PyMOL website](http://www.pymol.org)
- Helen M. Berman, John Westbrook, Zukang Feng, Gary Gilliland, T. N. Bhat, Helge 
Weissig, Ilya N. Shindyalov, Philip E. Bourne  
  The protein data bank  
  [Nucleic Acids Res. 28, 235–242 (2000)](https://doi.org/10.1093/nar/28.1.235)
- Gustaf Ahdritz, Nazim Bouatta, Sachin Kadyan, Lukas Jarosch, Daniel Berenberg, 
Ian Fisk, Andrew M. Watkins, Stephen Ra, Richard Bonneau, Mohammed AlQuraishi  
  OpenProteinSet: training data for structural biology at scale  
  [NeurIPS 37, Article 204 (2023)](https://openreview.net/forum?id=gO0kS0eE0F&noteId=ly7X3fS4uJ)  
  [AWS Open Data Registry](https://registry.opendata.aws/openfold/) (Accessed: 2025-08-05)
