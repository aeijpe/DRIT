# DRIT++

Large parts of the code are directly taken from the [original repo](https://github.com/HsinYingLee/DRIT).
Some small adjustments are made 
- to account for our dataset in `dataset.py`
- to be able to create the synthethis dataset in `create_fake_img.py`
- to evaluate the model on the LPIPS metric in `evaluate_model.py`

This repository is used as a basline in the project from [this repository](https://github.com/aeijpe/CrossModal-DRL).
Please refer to the README in the baselines folder in there to understand how to run this baseline

## Code structure

- In `results`, all checkpoints to the DRIT model for the different folds are stored.
- `src` contains the source code for the DRIT model.