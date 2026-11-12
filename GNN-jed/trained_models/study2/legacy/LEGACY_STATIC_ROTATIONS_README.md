# Legacy Archive: Static 5-Rotation Training Runs

This directory preserves the original static 5-rotation checkpoints, training logs, figures, and manifests for Study 2 prior to the migration to on-the-fly stochastic SO(3) data augmentation.

## Archived Artifacts

1. **GCN Clean Baseline (Static 5 Rotations)**
   - `best_gcn_clean.pth` (25 Epochs completed, Epoch 25 checkpoint, test accuracy 92.51%)
   - `training_metrics_gcn_clean.log`

2. **GAT Clean Checkpoint (Static 5 Rotations)**
   - `best_gat_clean_ep18_backup.pth` (Best validation loss 0.1470 at Epoch 18, test accuracy 93.04%)
   - `training_metrics_gat_clean.log.bak_ep22` (Full epoch-by-epoch training log up to Epoch 22)
   - `training_metrics_gat_clean.log` (Epoch 1 in-flight log)

3. **Validation & Visualization Artifacts**
   - `live_plots_GNN_clean_5obs.png` / `.html`
   - `error_percentage_by_perigalacticon_GAT_clean_5obs.png` / `.html`
   - Individual stream prediction plots (ID 1, 11, 53, 70, 74, 81)
   - `R_vs_t_6panels_rp1.png` / `.html`
   - `seeds_positions_unbiased.png` / `.html`
   - `spectral_analysis_GNN_clean_5obs.png` / `.html`
   - `study2_interactive_dashboard_clean_5obs.html`

4. **Model Registry & Manifests**
   - `weights/MODEL_WEIGHTS_REGISTRY.md`
   - `weights/weights_manifest.json`

## Rationale for Transition to Stochastic Rotations
- Under static 5 rotations, each epoch ingested all 5 rotations simultaneously ($32,825$ graphs, 513 batches of 64), leading to ~3h35m per epoch on 2 nodes (72 cores).
- Stochastic rotation cycles through 1 rotation per snapshot per epoch ($6,565$ graphs, 103 batches of 64), reducing epoch time to ~42m (or ~28m with AVX-512 bfloat16), a 5x speedup per epoch.
- All 5 rotations are uniformly sampled across epochs (10 times each over 50 epochs), eliminating intra-batch correlation while preserving full SO(3) rotational invariance.
- GCN baseline is scheduled for retraining under the identical stochastic regime in Step 7 to maintain an exact apples-to-apples comparison.
