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
```bibtex
@article{cgan2024,
  title={Conditional Generative Adversarial Network (Cgan) for Generating Building Load Profiles with Distributed Energy Resources (Der)},
  author={Li, Y., Dong, B., Qiu, Y.},
  year={2024},
  publisher={Preprint},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5025785},
  note={Preprint}
}
