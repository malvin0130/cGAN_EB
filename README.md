# CGAN for load generation with PV and EV
This repo includes the cGAN model and its training process as well as some generated samples of load profiles under different labels.

## Project Details
- [cGAN.py](./cGAN.py) - Model structure
- [train.py](./train.py) - Model training
- [utils.py](./utils.py) - Results analysis
- [`cGAN Generation`](https://github.com/malvin0130/cGAN_EB/tree/main/cGAN%20Generation) - Generated load profiles for different PV/EV labels

## Usage
1. Sample days of generated data from [cGAN Generation](https://github.com/malvin0130/cGAN_EB/tree/main/cGAN%20Generation) for the PV/EV status needed.
2. Aggregate to get complete profile

## Citation
If you use this code or data, please cite:
Li, Y., Dong, B., & Qiu, Y. (2025). Conditional Adversarial Network (cGAN) for Generating Building Load Profiles with Photovoltaics and Electric Vehicles. Energy and Buildings, 335, 115584. https://doi.org/10.1016/j.enbuild.2025.115584
