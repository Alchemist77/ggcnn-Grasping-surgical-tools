It is  extension of Generative Grasping CNN (GG-CNN) (https://github.com/dougsm/ggcnn).

The GGCNN is developed with simple encoder-decoder (kind of autoencoder) models. We implemented the model with different autoencoder models and also a script to evalute the performance of this network using Kinect v2.



##Autoencoder models 
*Denoising autoencoder

*Sparse Autoencoder

*Contractive Autoencoder

*Variational Autoencoder

*U-net


## Prerequisites
* Ubuntu 16.04

* Python 3

* pytorch

## run
For example,
```
python3 train_ggcnn.py --network sparse_ggcnn --dataset cornell --dataset-path /media/aal-ml/Ubuntu_data/olivia/catkin_ws/src/final_olivia --outdir output/ggcnn_sparse_model
```
For your environment,
```
python3 train_ggcnn.py --network path-to-the-network --dataset cornell --dataset-path path-to-the-dataset --outdir path-to-the-save folder
```


