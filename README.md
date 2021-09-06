# NerfingMVS
### [Project Page](https://weiyithu.github.io/NerfingMVS) | [Paper](https://arxiv.org/abs/2109.01129) | [Video](https://youtu.be/i-b5lPnYipA) | [Data](https://drive.google.com/drive/folders/1X_w57Q_MIFlI3lzhRt7Z8C5X9tNS8cg-?usp=sharing)
<br/>

> NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo  
> [Yi Wei](https://weiyithu.github.io/), [Shaohui Liu](http://b1ueber2y.me/), [Yongming Rao](https://raoyongming.github.io/), [Wang Zhao](https://github.com/thuzhaowang), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1)  
> ICCV 2021 (Oral Presentation)  

<p align='center'>
<img src="imgs/demo.gif" width='80%'/>
</p>

## Installation
- Pull NerfingMVS repo.
  ```
  git clone --recursive git@github.com:weiyithu/NerfingMVS.git
  ```
- Install python packages with anaconda.
  ```
  conda create -n NerfingMVS python=3.7
  conda activate NerfingMVS
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
  pip install -r requirements.txt
  ```
- We use COLMAP to calculate poses and sparse depths. However, original COLMAP does not have fusion mask for each view. Thus, we add masks to COLMAP and denote it as a submodule. Please follow https://colmap.github.io/install.html to install COLMAP in `./colmap` folder. 

## Usage
- Download 8 ScanNet scene data used in the paper here and put them under `./data` folder. We also upload final results and checkpoints of each scene here. 
- Run NerfingMVS
  ```
  sh run.sh $scene_name
  ```
  The whole procedure takes about 3.5 hours on one NVIDIA GeForce RTX 2080 GPU, including COLMAP, depth priors training, NeRF training, filtering and evaluation. COLMAP can be accelerated with multiple GPUs.You will get per-view depth maps in `./logs/$scene_name/filter`. Note that these depth maps have been aligned with COLMAP poses. COLMAP results will be saved in `./data/$scene_name` while others will be preserved in `./logs/$scene_name`

## Run on Your Own Data!
- Place your data with the following structure:
  ```
  NerfingMVS
  |───data
  |    |──────$scene_name
  |    |   |   train.txt
  |    |   |──────images
  |    |   |    |    001.jpg
  |    |   |    |    002.jpg
  |    |   |    |    ...
  |───configs
  |    $scene_name.txt
  |     ...
  ```
  `train.txt` contains names of all the images. Images can be renamed arbitrarily and '001.jpg' is just an example. You also need to imitate ScanNet scenes to create a config file in `./configs`. Note that `factor` parameter controls the resolution of output depth maps. You also should adjust `depth_N_iters, depth_H, depth_W` in `options.py` accordingly. 
- Run NerfingMVS without evaluation
  ```
  sh demo.sh $scene_name
  ```
  Since our work currently relies on COLMAP, the results are dependent on the quality of the acquired poses and sparse reconstruction from COLMAP.

## Acknowledgement
Our code is based on the pytorch implementation of NeRF: [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). We also refer to [mannequin challenge](https://github.com/google/mannequinchallenge). 
  
## Citation 
If you find our work useful in your research, please consider citing:
```
@inproceedings{wei2021nerfingmvs,
  author    = {Wei, Yi and Liu, Shaohui and Rao, Yongming and Zhao, Wang and Lu, Jiwen and Zhou, Jie},
  title     = {NerfingMVS: Guided Optimization of Neural Radiance Fields for Indoor Multi-view Stereo},
  booktitle = {ICCV},
  year = {2021}
}
```

