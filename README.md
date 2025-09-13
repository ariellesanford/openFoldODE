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
- **Data splits directory**: `data_splits/mini`
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
   bash ../save_intermediates/scripts/download_alphafold_params.sh openfold/resources
   ```

4. **Download PDB70 data**  
Change `DOWNLOAD_DIR` to your `{your_data_dir}` destination. Requires 113 GB storage.
   ```bash
   bash scripts/download_pdb70_mmcif_only.sh
   ```

5. **Generate Evoformer inputs**
   ```bash
   python helper_scripts/generate_evoformer_inputs.py --data-dir {your_data_dir} --splits-dir "data_splits/{your_splits_dir}"
   ```

6. **Generate blocks**  
   Generating all blocks for at least some proteins will allow you to use preliminary training to guide the protein folding evolution to mimic that of the evoformer, which may ultimately lead to better predictions.
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
`20250616_180845_full_ode_with_prelim_final_model.pt`  
The configuration settings, training progress, and preformance statistics for this model are at:
`20250616_180845_full_ode_with_prelim.txt` 
### Training

#### Quick Start with Sample Data
```bash
python training_runner.py --data-dir mini_data --splits-dir data_splits/mini --max-epochs 5 --no-prelim
```

#### General Usage
```bash
python training_runner.py --data-dir {your_data_dir} --splits-dir data_splits/{your_splits_dir} --max-epochs {num_epochs}
```


#### Training Runner Arguments and Configuration

The `training_runner.py` script accepts the following arguments:

**Core Arguments:**
- `--data-dir` (str, default: `/media/visitor/Extreme SSD/data`) - Base data directory containing complete_blocks and endpoint_blocks
- `--splits-dir` (str, default: `data_splits/jumbo`) - Directory containing train/val/test splits  
- `--max-epochs` (int, default: 200) - Maximum epochs for main training
- `device` (optional positional: `cpu` | `cuda`) - Force device selection (auto-detected if not specified) *Note: CPU training runs significantly slower than GPU training*

**Preliminary Training Options:**
- `--prelim` / `--no-prelim` (flag) - Enable/disable preliminary training (enabled by default)
- `--prelim-max-epochs` (int, default: 100) - Maximum epochs for preliminary training
- `--prelim-stride` (int, default: 4) - Stride for preliminary training blocks (e.g., 0→4→8→12→...→48)
- `--prelim-chunk-size` (int, default: 4) - Number of blocks to process at once during preliminary training

**Internal Model Configuration:**
- `num_layers`: 2 - Number of neural network layers
- `hidden_dim`: 128 - Hidden dimension size
- `time_embedding_dim`: 32 - Time embedding dimension
- `solver_method`: "dopri5" - ODE solver method
- `solver_rtol`: 1e-5, `solver_atol`: 1e-5 - ODE solver tolerances
- `adjoint`: True - Use adjoint method for memory-efficient backpropagation

**Internal Training Configuration:**
- `learning_rate`: 1e-4 - Initial learning rate
- `batch_size`: 1, `chunk_size`: 4 - Batch and chunk sizes
- `val_check_interval`: 50 - Validation check interval in steps
- `log_interval`: 10 - Logging interval in steps
- `save_interval`: 50 - Model save interval in steps
- `early_stopping_patience`: 20 - Epochs to wait before early stopping
- `early_stopping_min_delta`: 1e-4 - Minimum improvement for early stopping
- `scheduler_patience`: 10 - Epochs to wait before reducing learning rate
- `scheduler_factor`: 0.5 - Factor to reduce learning rate by
- `scheduler_min_lr`: 1e-6 - Minimum learning rate

## Troubleshooting  
If the error `$'\r': command not found` appears when you are trying to run any script try running:
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
