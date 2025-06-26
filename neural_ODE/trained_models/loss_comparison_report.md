# Loss Function Comparison Report

Generated: 2025-06-20 10:27:02

## Experiments Overview

| Experiment | Loss Strategy | Status | Output Directory |
|------------|---------------|--------|-----------------|
| Default Loss (Full MSA) | `default` | ✅ Success | `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models` |
| Weighted Row Loss (5x First Row) | `weighted_row` | ✅ Success | `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models` |
| Single Row Loss (Structure-Focused) | `single_row` | ✅ Success | `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models` |

## Loss Strategy Details

### Default Loss
- Uses full MSA representation: `F.mse_loss(pred_m, target_m)`
- All MSA rows weighted equally
- Most computationally expensive

### Weighted Row Loss
- First MSA row weighted 5x more than others
- Normalized weighting (weights sum to 1.0)
- Balances structure importance with full MSA training

### Single Row Loss
- Only uses first MSA row: `F.mse_loss(pred_m[0, :, :], target_m[0, :, :])`
- Most structure-focused approach
- Fastest training and most memory efficient

## Results Analysis

Compare the following metrics across experiments:

1. **Training Loss**: Check final training losses in each experiment's log
2. **Validation Loss**: Compare validation performance
3. **Structure Quality**: Run structure prediction tests
4. **Training Speed**: Compare epochs/hour and memory usage
5. **Convergence**: Check how quickly each approach reaches good performance

## Files to Check

### Default Loss (Full MSA)
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250619_161154_loss_comparison_default_full_msa.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250619_161154_loss_comparison_default_full_msa_best.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250619_161154_loss_comparison_default_full_msa_automation.txt`

### Weighted Row Loss (5x First Row)
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250619_161154_loss_comparison_weighted_row_5x_first.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250619_161154_loss_comparison_weighted_row_5x_first_best.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250619_161154_loss_comparison_weighted_row_5x_first_automation.txt`

### Single Row Loss (Structure-Focused)
- Training log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250619_161154_loss_comparison_single_row_structure_focused.txt`
- Model checkpoint: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/20250619_161154_loss_comparison_single_row_structure_focused_best.pt`
- Automation log: `/home/visitor/PycharmProjects/openFold/neural_ODE/trained_models/automation_logs/20250619_161154_loss_comparison_single_row_structure_focused_automation.txt`

