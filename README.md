
# GDR-Net

Official PyTorch implementation of **GDR-Net: Complementary Graph Disentanglement and Feature Recombination for Pediatric EEG Seizure Detection and Subtype Classification**.

GDR-Net integrates temporal–spectral EEG encoding, complementary graph disentanglement, and label-compatible feature recombination for automated seizure analysis.

The framework supports two separately trained tasks:
- **Seizure detection:** Binary classification of seizure and background EEG windows.
- **Seizure subtype classification:** Conditional classification of absence (ABSZ), tonic (TNSZ), and tonic–clonic (TCSZ) seizures.

## Requirements

Python 3.10+ and PyTorch 2.2+ are recommended.

Install the required packages:

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with your CUDA environment.

## Datasets

Three EEG datasets were used in this study:

- **CHB-MIT:** [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) — publicly available.
- **TUSZ v2.0.6:** [NEDC](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/) — requires approved access.
- **WHCH:** Wuhan Children's Hospital — restricted clinical dataset.

CHB-MIT and TUSZ were used for seizure detection development, while subtype classification used TUSZ alone. WHCH was used for retrospective external evaluation.

Raw EEG recordings, processed signals, patient-level metadata, and trained checkpoints are not included in this repository.

## Data Preparation

The preprocessing scripts standardize EEG recordings to 18 bipolar channels at 256 Hz. The analysis uses 8-second windows with a 4-second stride.

Run the following commands from the repository root:

```bash
# CHB-MIT
python scripts/prepare_chbmit.py \
    --chbmit_root /path/to/CHB-MIT \
    --output_root data/chbmit_processed

# TUSZ
python scripts/prepare_tusz.py \
    --tusz_root /path/to/TUSZ \
    --output_root data/tusz_processed

# WHCH (authorized users only)
python scripts/prepare_whch.py \
    --whch_root /path/to/WHCH \
    --output_root data/whch_processed
```

The expected processed dataset directories are:

```text
data/
├── chbmit_processed/
├── tusz_processed/
└── whch_processed/
```

## Training

The repository includes GDR-Net and four baseline models. Training configurations are provided in `configs/`.

Generate the experimental configurations:

```bash
python scripts/manage_experiments.py prepare --data-root data
```

Run the experiments:

```bash
python scripts/run_experiments.py run \
    --manifest runs/plan/paper.json \
    --gpus 0
```

Multiple GPU IDs can be provided to execute independent experiments in parallel.

Check training progress:

```bash
python scripts/run_experiments.py status \
    --manifest runs/plan/paper.json
```

The default study configuration includes three random seeds and patient-disjoint source development splits.

## External Evaluation

After source training is complete, generate the external evaluation manifest:

```bash
python scripts/manage_experiments.py evaluate \
    --manifest runs/plan/paper.json
```

Run external evaluation:

```bash
python scripts/run_experiments.py run \
    --manifest runs/evaluation/paper_external.json \
    --gpus 0
```

Summarize the results:

```bash
python scripts/manage_experiments.py summarize \
    --manifest runs/evaluation/paper_external.json
```

Evaluation uses source-selected checkpoints and operating settings without target-specific model fitting or threshold adjustment.

WHCH data are required to reproduce the external evaluation reported in the paper.

## Baseline Models

Four baseline architectures were adapted to the study's EEG preprocessing, training, and evaluation protocol.

| Model | Reference Repository |
| --- | --- |
| EEGNet | [ARL EEGModels](https://github.com/vlawhern/arl-eegmodels) |
| CNN–LSTM | Study-specific implementation included in this repository |
| EEG Conformer | [EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer) |
| DCRNN | [EEG-GNN-SSL](https://github.com/tsy935/eeg-gnn-ssl) |

All four baseline implementations are included in `src/models/`.

These are study-specific adaptations and should not be interpreted as exact reproductions of the original training pipelines. Third-party references and applicable licensing notices are provided in `THIRD_PARTY_NOTICES.md`.

## License

The original GDR-Net project code is intended to be released under the MIT License, subject to confirmation of the relevant rights and third-party licensing obligations.

See `LICENSE` and `THIRD_PARTY_NOTICES.md` for details.

## Citation

If you use this code in your research, please cite:

**GDR-Net: Complementary Graph Disentanglement and Feature Recombination for Pediatric EEG Seizure Detection and Subtype Classification.**

Publication details will be added after acceptance.
