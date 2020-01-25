# Dirty-Gancing
This repository contains the code and models for the Dirty GANcing Visium workshop at AMLD 2020.

## Workshop setup

Running the code of this repository is, for the most part, computionally expensive and it is best to have GPU access. For the workshop users, we have prepared the notebook so that you can easily load them into Google Colab and enjoy the **free GPU services** offered by Google.

1) Open [Google Colab](https://colab.research.google.com/) and sign in with your Google account or create a new one.

2) Click on `File` and then `Open notebook...`. Select the **Github** tab and type in look for **VisiumCH** to find this repository and select the notebook you would like to run.

3) Once you are in the notebook, click on **Runtime**, **Change runtime type** and then select **GPU** as hardware accelerator.

4) Finally, click on **Connect** and you should be ready!


## Requirements
For users outside the workshopwho would like to experiment more with the repo, here are a few installation steps:

Create a virtual environnement and install the dependencies:
```bash
python -m venv env
source env/bin/activate
pip install -r requirements_1st.txt
pip install -r requirements_2nd.txt
```

## Basic usage

1)  Put your source video (an `.mp4` file) in `./data/sources/{source_folder_name}/` 
and then run:
```bash
python src/data_preparation/prepare_source.py -s data/sources/{source_folder_name}
```
2) Put your target video (an `.mp4` file) in `./data/targets/{target_folder_name}/` 
and then run: 
```bash
python src/data_preparation/prepare_target.py -s data/targets/{target_folder_name}
```

3) Train the pose2vid model by running:
```bash
python src/GANcing/train_pose2vid.py -t data/targets/{target_folder_name} -r {run_name}
```
The training can be monitored:
* Using Tensorboard by running `tensorboard --logdir ./checkpoints`
* In a basic html webpage by running `python -m http.server` in `./checkpoints/{run_name}/web`

4) If the training is stopped for any reason, you can restart it from the last
checkpoint by using the same command as in step 3.

5) Once the training is finished or you stopped it, you can perform the Pose Normalization with:
```bash
python src/data_postprocessing/normalization.py -s data/sources/{source_folder_name} -t data/targets/{target_folder_name}
```

6) Finally, you can perform the pose transfer with:
```bash
python src/data_postprocessing/transfer.py -s data/sources/{source_folder_name} -r {run_name}
```

7) You can visualize the results as a gif by running:
```bash
python src/data_postprocessing/make_gif.py -s data/sources/{source_folder_name} -r results/{run_name}
```

## Extended notice

This extended instructions are for users willing to add Face Enhancement to their model

8) Start by creating the necessary files with:
```bash
python src/face_enhancer/prepare_face_enhancer_data.py -t data/targets/{target_folder_name} -r {run_name}
```

9) Train the Face Enhancer with:
```bash
python src/face_enhancer/train_face_enhancer -t data/targets/{target_folder_name} -r {run_name}
```

10) Once you are satisfied with the training results, you can stop it and perform the face_enhancement with:
```bash
python src/face_enhancer/run_face_enhancer.py -s data/sources/{source_folder_name} -t data/targets/{target_folder_name} -r {run_name}
```
This will create a new results folder in `./results/{run_name}_enhanced`, which you can use as a substitue in `make_gif.py`

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