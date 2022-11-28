# YoloV3 Implemented in TensorFlow 2.0

## Usage
### Installation

#### Conda (Recommended)
```
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov3-tf2-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov3-tf2-gpu
```

#### Pip
```
pip install -r requirements.txt
```

### Nvidia Driver (For GPU)
```
# Ubuntu 18.04
sudo apt-add-repository -r ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```

### Dataset
```
./ship1000_data/  하위에 폴더 위치

# Training Dataset
./ship1000_data/train/images
./ship1000_data/train/labels

# Validation Dataset
./ship1000_data/validation/images
./ship1000_data/validation/labels

```


### Detection

```
# validation image
python detect.py --image ./data/meme.jpg

# custom image
python detect.py --isValidation False --image ./data/objects365_ship.jpg

```

### Training

``` 
python train.py 

```

