#!/bin/bash
sudo chown -R jie.he2:jie.he2 /opt/conda/
export PIP_REQUIRE_VIRTUALENV=false
pip3 install h5py
pip3 install ftfy regex tqdm
export https_proxy=http://proxy-bj.nioint.com:8080
pip3 install git+https://github.com/openai/CLIP.git
python main.py --cfg ./config/ImageNet_LT/clip_A_rn50.yaml
python main.py --cfg ./config/ImageNet_LT/clip_B_rn50.yaml