# Third-party references and notices

This release includes project-specific implementations of published model
architectures. The links below identify the original implementations and their
upstream license notices; they do not assign those licenses to independently
written implementations solely because the architectures are similar.

## EEGNet

- Repository: https://github.com/vlawhern/arl-eegmodels
- Upstream license: https://github.com/vlawhern/arl-eegmodels/blob/master/LICENSE.txt
- The upstream notice specifies CC0 1.0 for portions of the work and Apache
  License 2.0 for other portions. The complete upstream notice governs any
  upstream code reused or redistributed.
- Local implementation: `src/models/baselines.py` (`EEGNetBaseline`).

## EEG Conformer

- Repository: https://github.com/eeyhsong/EEG-Conformer
- Upstream license: https://github.com/eeyhsong/EEG-Conformer/blob/main/LICENSE
- The upstream repository is distributed under GPL-3.0. Preserve applicable
  upstream notices and follow its terms for any reused upstream code.
- Local implementation: `src/models/eeg_conformer.py`.

## EEG DCRNN

- Repository: https://github.com/tsy935/eeg-gnn-ssl
- Upstream license: https://github.com/tsy935/eeg-gnn-ssl/blob/main/LICENSE
- Upstream notice: MIT License, Copyright (c) 2021 Siyi Tang. Its copyright and
  permission notice must accompany copies or substantial portions of that code.
- Local implementation: `src/models/dcrnn.py`.

## CNN–LSTM

The CNN–LSTM baseline in `src/models/baselines.py` is implemented for this study.
No third-party repository is presented as its official implementation.

## Release license status

The original code developed for GDR-Net is released under the MIT License. Any incorporated third-party code remains subject to its applicable upstream license and attribution requirements. These notices
document upstream sources and do not grant a license for the entire GDR-Net
project. Dependency packages retain their respective licenses.
