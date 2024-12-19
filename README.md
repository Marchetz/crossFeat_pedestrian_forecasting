# crossFeat_pedestrian_forecasting
Official Pytorch code for the paper "CrossFeat: Semantic Cross-modal Attention for Pedestrian Behavior Forecasting" (t-IV).

#in train.py, set configuration for Comet,  set api_key, project_name and workspace


## Installation
To install the required packages, create a conda environment using this command: 
```bash
conda create -n <name_env> python=3.7
```

Pytorch installation:
Then you can install PyTorch 1.9.1 choosing your cude version (this code has been tried with cuda 11) required in this repo:
```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```


## Train the Model
To train a model by scratch, utilize the script train.py

For example, use this command:
```bash
python train.py
```

## Test the Model
In a pretrained folder, there are the pretrained weights of the models.

To test the model, use this command:
```bash
python eval.py
```



## Acknowledgment
The code is based on repo "https://github.com/vita-epfl/hybrid-feature-fusion" of paper "Pedestrian Stop and Go Forecasting with Hybrid Feature Fusion" (Dongxu Guo , Taylor Mordan and Alexandre Alahi). 
We thank the authors for making the code available and usable.
