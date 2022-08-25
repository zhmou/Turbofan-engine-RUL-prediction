# Turbofan engine RUL prediction
A reproduction of this paper by using PyTorch: [*Machine Remaining Useful Life Prediction via an Attention-Based Deep Learning Approach*](https://personal.ntu.edu.sg/xlli/publication/RULAtt.pdf)  
for TensorFlow implementation, please visit the [original author's repository](https://github.com/ZhenghuaNTU/RUL-prediction-using-attention-based-deep-learning-approach)


## Table of Contents
- [Requirements](#requirements)
- [Usage](#usage)
    - [Download](#download)
    - [Open project with your IDE](#open-project-with-your-ide)
    - [Run main.py](#run-mainpy)
- [Network Architecture](#network-architecture)
- [Eval Metrics](#eval-metrics)
- [Result](#result)

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

This paper **only** tested two sub-datasets of CMAPSS(FD001 and FD004). if you are interested in other datasets, **don't forget to normalizing the raw data by trying preprocess.py**

### Run main.py
It will take up about 2~3 minutes to load the whole dataset, don't worry.  
When the output from the console looks like this, **congratulations**, the network is training now:  
![image](https://user-images.githubusercontent.com/43105172/165487346-7618b07c-3f06-448f-8000-ba80eafbe93d.png)  
The program will save the model parameters at <code>./checkpoints/</code> automaticlly during every iteration when it found a better result.

<ins>**Based on the feedback, you will need to manually create this folder (<code>./checkpoints/</code>) in the current path to avoid reporting an error.**</ins>

After 10 iterations(32 epochs per iteration), best result of each iteration under eval metrics would save to a txt like this:  
![image](https://user-images.githubusercontent.com/43105172/165488259-6da54a06-0aae-4322-ab92-b5ca8fa5e0d3.png)

## Network Architecture
![image](https://user-images.githubusercontent.com/43105172/165488689-dfd63dcd-84a6-4c01-bd67-cbfb2d19b00c.png)  
The network can mainly spilt into two parts:  
The left one takes a 30(windows size, or time step, default by 30) * 17(sensory nums) sequential data as the inputs of one sample. Firstly, it would be sent into LSTM to output a 30 * 50 feature map. Then, a very simplified attention mechanism would be performed, It will caculate the weights of each particular feature and get an attention matrix:  
![image](https://user-images.githubusercontent.com/43105172/165677077-c3850bb5-9410-4972-af9d-dc0f928c83d9.png)  
The attention matrix will make a dot product with the feature map and flatten to be a 1D vector of length 1500. After 2 linear layers(with ReLU, dropout, etc.), we finally get a 1D vector of length 10.

Take a look at the right-side part. for every column of each sample, there are two handcrafted features can be extracted: mean value and trend coefficient(or you can say the slope of the line fitted to these 30 points in one column). Since each sample has 17 columns, we can obtain a 1D vector of length 34. As before, after a linear layer, we get a vector of length 10.

We concatenate these two vectors and get a 1D vector of length 20. The finally thing is to pass through a output layer and get our predicted RUL value. (Note that the label of the dataset is normalized by dividing by the max_rul. )


## Eval Metrics
- RMSE: root of MSE
- Score:  
![image](https://user-images.githubusercontent.com/43105172/165489175-61bf3a63-e56b-4efe-9986-c003b1b56c4e.png)
>&emsp;&emsp;This scoring function penalizes late predictions (too late to perform maintenance) more than early predictions (no big harms although it could waste maintenance resources). This is in line with the risk adverse attitude in aerospace industries. However, there are several drawbacks with this function. The most significant drawback being a single outlier (with a much late prediction) would dominate the overall performance score, thus masking the true overall accuracy of the algorithm. Another drawback is the lack of consideration of the prognostic horizon of the algorithm. The prognostic horizon assesses the time before failure which the algorithm is able to accurately estimate the RUL value within a certain con- fidence level. Finally, this scoring function favors algorithms which artificially lowers the score by underestimating RUL. Despite all these shortcomings, the scoring function is still used in this paper to provide comparison results with other methods in literature.  

~[*Deep Convolutional Neural Network Based Regression Approach for Estimation of Remaining Useful Life*](https://oar.a-star.edu.sg/storage/r/r3zk8v8r78/dasfaa2016-014-final-v1.pdf)

## Result
The original paper results:  
![image](https://user-images.githubusercontent.com/43105172/165681622-ebd1fe3b-9337-4839-b972-51f704511aae.png)  
Guess it's the average or median value of the best results of 10 iterations.  

My results(10 best result of every iteration):  
**FD001**:  
![image](https://user-images.githubusercontent.com/43105172/165488259-6da54a06-0aae-4322-ab92-b5ca8fa5e0d3.png)  
**FD004**:  
![image](https://user-images.githubusercontent.com/43105172/165688530-9b4ccd65-7384-419b-b0a7-5dbb5e2c0dec.png)  

On FD004, the results reproduced using PyTorch differ slightly from those of the original paper, which can be seen is that in my results. The 10 best results are more **spread out** compared to original authors' paper:
![image](https://user-images.githubusercontent.com/43105172/165689516-18251719-709c-48d0-a318-af35beda31d8.png)  

I am trying to figure out the cause of this problem, if you have a good idea please [post a issue](https://github.com/zhmou/zhmou.github.io/issues/new) and let me know, thanks gratefully!
