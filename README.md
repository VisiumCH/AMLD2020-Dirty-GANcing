# Dirty-Gancing
This repository contains the code and models for the Dirty GANcing Visium workshop at AMLD 2020.

## Requirements
If you need to train the pose2vid network, it is best to have GPU capabilities. Most of the rest of the code could probably run on cpu.

Create a virtual environnement and install the dependencies:
```bash
python -m venv env
source env/bin/activate
pip install -r requirements_1st.txt
pip install -r requirements_2nd.txt
```

## Citations

This repo is an adaptation of the paper [Everybody Dance Now](https://arxiv.org/pdf/1808.07371.pdf). If you intend to use it for anything, please consider citing the original authors in your paper:

```
@article{chan2018everybody,
  title={Everybody dance now},
  author={Chan, Caroline and Ginosar, Shiry and Zhou, Tinghui and Efros, Alexei A},
  journal={arXiv preprint arXiv:1808.07371},
  year={2018}
}
```

The code in this repo is an adapted and corrected for the AMLD workshop, and is orginally based on [CUHKSZ-TQL's repo](https://github.com/CUHKSZ-TQL/EverybodyDanceNow_reproduce_pytorch). 

It also borrows heavily from :
* [pix2pixHD](https://github.com/NVIDIA/pix2pixHD) 
* [pytorch_Realtime_Multi-Person_Pose_Estimation](https://github.com/last-one/Pytorch_Realtime_Multi-Person_Pose_Estimation)