# Loss Function Comparison Experiment Report

Generated: 2025-06-20 18:57:58

## Experiment Overview

This automated comparison tested different loss function strategies:

1. **Loss Comparison: Default** - ✅ SUCCESS
   - Loss function: `default`
   - Max residues: 450
   - Prelim chunk size: 2
   - Prelim block stride: 4
   - Experiment name: `20250620_185615_loss_comparison_default`

2. **Loss Comparison: Weighted Row** - ✅ SUCCESS
   - Loss function: `weighted_row`
   - Max residues: 450
   - Prelim chunk size: 2
   - Prelim block stride: 4
   - Experiment name: `20250620_185642_loss_comparison_weighted_row`

3. **Loss Comparison: Single Row (Structure Focused)** - ✅ SUCCESS
   - Loss function: `single_row`
   - Max residues: 450
   - Prelim chunk size: 2
   - Prelim block stride: 4
   - Experiment name: `20250620_185709_loss_comparison_single_row_structure_focused`

4. **Enhanced Preliminary Training** - ✅ SUCCESS
   - Loss function: `default`
   - Max residues: 400
   - Prelim chunk size: 3
   - Prelim block stride: 2
   - Experiment name: `20250620_185736_enhanced_preliminary_training_default`

## Experiment Configurations

### Base Configuration
- **data_dirs**: ['/media/visitor/Extreme SSD/data/complete_blocks', '/media/visitor/Extreme SSD/data/endpoint_blocks']
- **splits_dir**: /home/visitor/PycharmProjects/openFold/neural_ODE/data_splits/1fv5
- **device**: cuda
- **epochs**: 10000
- **learning_rate**: 0.001
- **reduced_cluster_size**: 64
- **hidden_dim**: 64
- **integrator**: rk4
- **use_fast_ode**: False
- **use_amp**: True
- **output_dir**: /home/visitor/PycharmProjects/openFold/neural_ODE/trained_models
- **lr_patience**: 3
- **lr_factor**: 0.5
- **min_lr**: 1e-06
- **early_stopping_patience**: 10
- **early_stopping_min_delta**: 0.0001
- **max_time_hours**: 0.005
- **aggressive_cleanup**: True
- **enable_preliminary_training**: True
- **prelim_data_dir**: /media/visitor/Extreme SSD/data/complete_blocks
- **prelim_max_epochs**: 2

### Experiment-Specific Variations

#### Loss Comparison: Default
- Loss function: `default`
- Max residues: 450
- Prelim chunk size: 2
- Prelim block stride: 4

#### Loss Comparison: Weighted Row
- Loss function: `weighted_row`
- Max residues: 450
- Prelim chunk size: 2
- Prelim block stride: 4

#### Loss Comparison: Single Row (Structure Focused)
- Loss function: `single_row`
- Max residues: 450
- Prelim chunk size: 2
- Prelim block stride: 4

#### Enhanced Preliminary Training
- Loss function: `default`
- Max residues: 400
- Prelim chunk size: 3
- Prelim block stride: 2

## Analysis Guidelines

To compare results, examine:

1. **Training Loss**: Check final training losses in each experiment's log
2. **Validation Loss**: Compare validation performance
3. **Structure Quality**: Run structure prediction tests
4. **Training Speed**: Compare epochs/hour and memory usage
5. **Convergence**: Check how quickly each approach reaches good performance
6. **Preliminary Training**: Compare preliminary vs main training performance

## Files to Check

### Loss Comparison: Default
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185615_loss_comparison_default.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185615_loss_comparison_default_final_model.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250620_185615_loss_comparison_default_automation.txt`

### Loss Comparison: Weighted Row
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185642_loss_comparison_weighted_row.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185642_loss_comparison_weighted_row_final_model.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250620_185642_loss_comparison_weighted_row_automation.txt`

### Loss Comparison: Single Row (Structure Focused)
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185709_loss_comparison_single_row_structure_focused.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185709_loss_comparison_single_row_structure_focused_final_model.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250620_185709_loss_comparison_single_row_structure_focused_automation.txt`

### Enhanced Preliminary Training
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185736_enhanced_preliminary_training_default.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250620_185736_enhanced_preliminary_training_default_final_model.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250620_185736_enhanced_preliminary_training_default_automation.txt`

