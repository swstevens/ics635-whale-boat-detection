# SatViT

The SatViT-V2 encoder used by `notebooks/train_chip_classifier.ipynb`. This code comes from Joe
Behrens's ICS 605 Oahu flood project, unchanged:

- `models.py`: builds the encoder with the SatViT-V2 dimensions. The flood decoder in this file
  isn't used here.
- `SatViT/`: the transformer implementation by antofuller
  ([github.com/antofuller/SatViT](https://github.com/antofuller/SatViT)), with its license.

## Weights

`SatViT_V2.pt` is too large for GitHub, so it's hosted on Google Drive (UH account required). Put
it at `checkpoints/SatViT_V2.pt`. The notebook skips SatViT if the file isn't there.

The flood project listed these three links without saying which is which. One is `SatViT_V2.pt`;
the other two are that project's `final_project_encoder.pth` and `final_project_decoder.pth`, which
aren't needed here.

- <https://drive.google.com/file/d/1eTs08RpSIHsnn5nrwd354qOBhvf6-GbQ/view?usp=drive_link>
- <https://drive.google.com/file/d/1Gaojkrm8rRK7fXBCirungBu2es8oPicl/view?usp=drive_link>
- <https://drive.google.com/file/d/1zvD8JlA2Z1hGk_G59yCLsPwFO9LvYhZ8/view?usp=drive_link>
