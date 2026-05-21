# Reference analysis notebooks

These notebooks are provided as reference analysis code for the accepted paper figures. They are not intended to be a standalone reproduction package.

Model weights, LoRA adapters, datasets, and intermediate hidden-state files are not included. Paths under `data/`, `adapters/`, and `outputs/` are illustrative local paths used by the analysis scripts.

## Notebooks

- `cka_difference_heatmap.ipynb`: plots the fixed-scale CKA difference heatmap, `ours - baseline`.
- `layerwise_mae_arc_challenge.ipynb`: computes layer-wise mean attention entropy over preprocessed ARC-Challenge prompts. Here `MAE` means mean attention entropy, not mean absolute error.
- `layerwise_representation_metrics.ipynb`: computes effective rank, adjacent-layer alignment, log-det covariance, and a Gaussian pseudo-MI/log-det diagnostic to the final layer.
- `svcca_layer_similarity_heatmaps.ipynb`: computes SVCCA heatmaps for baseline, ours, and baseline-vs-ours layer comparisons. In the cross heatmap, rows are baseline layers and columns are ours layers.
- `unembedding_access_ratio.ipynb`: computes decoder-layer access to the shared base-model unembedding subspace before the final model norm.

## Expected local layout

```text
data/
adapters/baseline/
adapters/ours/
outputs/
```

The ARC-Challenge file used by `layerwise_mae_arc_challenge.ipynb` is expected to be a preprocessed instruction/input/output-style JSON file. Raw ARC dataset preprocessing is not included.
