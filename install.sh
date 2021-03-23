#!/bin/sh
#apt-get install git -y
apt-get update
apt-get install vim -y
apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 -y
apt-get install wget -y
wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
bash Anaconda3-2020.02-Linux-x86_64.sh -b

git clone https://github.com/circulosmeos/gdown.pl.git
./gdown.pl/gdown.pl https://drive.google.com/file/d/1Fe8iISLdcZ_COidYyJ2utg8ODVi_S5yt/view VoxCeleb1Archive
tar - xvf VoxCeleb1Archive VoxCeleb1/
rm VoxCeleb1Archive

conda init bash
source ~/.bashrc
conda create -n marionet python=3.8 -y
conda activate marionet

mkdir checkpoints
mkdir generated_images

pip install wandb
