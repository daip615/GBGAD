# GBGAD

This directory contains the standalone implementation of GBGAD for
unsupervised anomaly detection. The implementation learns latent sample
representations with an autoencoder, constructs and periodically refreshes an
adaptive granular-ball graph in the latent space, propagates regional context
with a graph convolutional network, and predicts sample-level anomaly scores
from the concatenation of the sample representation and its assigned
granular-ball graph embedding.

The graph distance is `d_mn = ||c_m - c_n||_2`, the Euclidean distance between
granular-ball centers. Neighborhood selection, Gaussian edge weights, and
isolation scores use this distance without adding the ball radii. Radii remain
in the ball node features and within-ball offset normalization.

## Files

- `gbgad/balls.py`: adaptive granular-ball construction based on recursive
  two-means splitting and the sum-of-distances criterion.
- `gbgad/model.py`: autoencoder, granular-ball graph convolution, and scoring
  network.
- `gbgad/losses.py`: reconstruction, compactness, and pseudo-label losses.
- `gbgad/trainer.py`: structure refresh, teacher-score construction, joint
  optimization, and final inference.
- `gbgad/data.py`: MAT, NPZ, and CSV dataset readers.
- `experiment_data.py`: experiment loader that prevents labels from entering
  the feature matrix or training process.
- `export_gbgad_scores.py`: batch entry point for exporting sample-level
  anomaly scores.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Data

The batch runner accepts a single `.npz`, `.mat`, or `.csv` file, or a
directory containing such files. For ADBench NPZ files, the expected arrays are
`X` and `y`. Labels are used only after training to report evaluation metrics
and are never passed to GBGAD. For legacy MAT files whose `trandata` matrix
stores labels in its final column, that column is explicitly removed from the
training features.

## Running GBGAD

From this directory, run:

```bash
python export_gbgad_scores.py --data "path/to/datasets" --output-dir "outputs/GBGAD" --device cuda
```

Use `--device cpu` when CUDA is unavailable. The default structural parameters
are latent dimension 8, granular-ball graph neighborhood size 5, and minimum
granular-ball size 5. All datasets use the same defaults unless parameters are
explicitly supplied on the command line.

Each output CSV contains `index`, `score`, and `ball_id`; a `label` column is
included when labels are available. The batch summary is saved as
`gbgad_summary.csv` in the output directory.
