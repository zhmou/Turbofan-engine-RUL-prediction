# Turbofan engine RUL prediction
A reproduction of this paper by using PyTorch: [Machine Remaining Useful Life Prediction via an Attention-Based Deep Learning Approach](https://personal.ntu.edu.sg/xlli/publication/RULAtt.pdf)  
for TensorFlow implementation, please visit the [original author's repository](https://github.com/ZhenghuaNTU/RUL-prediction-using-attention-based-deep-learning-approach)

## Requirements
[![Python badge](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch badge](https://img.shields.io/badge/PyTorch-1.11.0-green.svg)](https://pytorch.org/)
[![NumPy badge](https://img.shields.io/badge/Numpy-1.22.2-yellow.svg)](https://numpy.org/)  

The package version listed above is the version I used during my development, but you can still try other versions.

## Usage
### Download
```
git clone https://github.com/zhmou/Turbofan-engine-RUL-prediction.git
cd ./Turbofan-engine-RUL-prediction
```
### Open project with your IDE
Click main.py and modify the load path of the dataset to suit your needs.  
![image](https://user-images.githubusercontent.com/43105172/165482739-6f4f4fc6-6dca-4a08-9ef4-a84b206e9e4f.png)

If you load a specific dataset, you need to change the max_rul value in turbofandataset.py:
![image](https://user-images.githubusercontent.com/43105172/165485985-2ab4f835-6006-45be-bee6-530e83ce2c05.png)  
(for FD001, the value of max_rul is 130 and 150 for FD004. Please refer to [another paper](https://oar.a-star.edu.sg/storage/r/r3zk8v8r78/dasfaa2016-014-final-v1.pdf))

This paper **only** tested two sub-datasets of CMAPSS(FD001 and FD004). if you are interested in other datasets, don't forget to normalizing the raw data by trying preprocess.py

### Run main.py
It will take up about 2~3 minutes to load the whole dataset, don't worry.  
When the output from the console looks like this, **congratulations**, the network is training now:  
![image](https://user-images.githubusercontent.com/43105172/165487346-7618b07c-3f06-448f-8000-ba80eafbe93d.png)  
The program will save the model parameters automaticlly during every iteration when it found a better result.

Ater 10 iterations(32 epochs per iteration), best result under eval metrics would save to a txt like this:
![image](https://user-images.githubusercontent.com/43105172/165488259-6da54a06-0aae-4322-ab92-b5ca8fa5e0d3.png)

## Network Architecture
![image](https://user-images.githubusercontent.com/43105172/165488689-dfd63dcd-84a6-4c01-bd67-cbfb2d19b00c.png)

## Eval metrics
- RMSE: root of MSE
- Score:  
![image](https://user-images.githubusercontent.com/43105172/165489175-61bf3a63-e56b-4efe-9986-c003b1b56c4e.png)
