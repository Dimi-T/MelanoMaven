#!/bin/bash

echo Fetching Updates...
exec > /dev/null
sudo apt --fix-broken install && apt-get -y update && sudo apt-get -y upgrade
exec > /dev/tty
echo Done!

echo Installing Dependencies...
exec > /dev/null
sudo apt-get -y install python3.10
sudo apt-get -y install python3-pip
sudo apt-get -y install libmagickwand-dev
sudo apt-get -y install libmagickcore-dev
exec > /dev/tty
echo Done!


echo Installing Cuda...
exec > /dev/null
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get -y install cuda
sudo rm -f cuda-keyring_1.0-1_all.deb
exec > /dev/tty
echo Done!

echo Installing Python Packages...
exec > /dev/null
pip3 install -qq -r requirements.txt
exec > /dev/tty
echo Done!