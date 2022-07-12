# Social Navigation



##  iGibson

iGibson is a simulation environment providing fast visual rendering and physics simulation based on Bullet. iGibson is equipped with fifteen fully interactive high quality scenes, hundreds of large 3D scenes reconstructed from real homes and offices, and compatibility with datasets like CubiCasa5K and 3D-Front, providing 8000+ additional interactive scenes. Some of the features of iGibson include domain randomization, integration with motion planners and easy-to-use tools to collect human demonstrations. With these scenes and features, iGibson allows researchers to train and evaluate robotic agents that use visual signals to solve navigation and manipulation tasks such as opening doors, picking up and placing objects, or searching in cabinets.


### Citation

```
@article{shenigibson,
  title={iGibson, a Simulation Environment for Interactive Tasks in Large Realistic Scenes},
  author={Shen*, Bokui and Xia*, Fei and Li*, Chengshu and Mart{\'i}n-Mart{\'i}n*, Roberto and Fan, Linxi and Wang, Guanzhi and Buch, Shyamal and D’Arpino, Claudia and Srivastava, Sanjana and Tchapmi, Lyne P and  Vainio, Kent and Fei-Fei, Li and Savarese, Silvio},
  journal={arXiv preprint arXiv:2012.02924},
  year={2020}
}
```

## Installation
sudo apt install nvidia-cuda-toolkit

### create conda environment
(lookup: install conda)

conda update -y conda

conda create -y -n socnav python=3.8

source activate socnav

### (pyrvo)
(git clone https://github.com/sybrenstuvel/Python-RVO2/)

(python setup.py build)

(pip install setuptools)

pip install Cython

cd social-navigation

cd iGibson

pip install -e .

pip install stable-baselines3[extra]

pip install protobuf==3.20

### installing scenes, etc.
python -m gibson2.utils.assets_utils –download_assets

