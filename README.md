It is different application using Generative Grasping CNN (GG-CNN) (https://github.com/dougsm/ggcnn).

The GGCNN is developed with simple encoder-decoder (kind of autoencoder) models. We implemented the model with different autoencoder models and also a script to evalute the performance of this network using Kinect v2. Also, we make several scripts for training different models conveniently.  



## Autoencoder models 
* Denoising autoencoder

* Sparse Autoencoder

* Contractive Autoencoder

* Variational Autoencoder

* U-net


## Prerequisites
* Ubuntu 16.04

* Python 3

* pytorch

## run
For example,
```
python3 train_ggcnn.py --network ggcnn --dataset cornell --dataset-path /media/aal-ml/Ubuntu_data/olivia/catkin_ws/src/final_olivia --outdir output/ggcnn_model
```
```
python3 train_ggcnn_vit.py --network vit_ggcnn --dataset cornell --dataset-path /media/aal-ml/Ubuntu_data/olivia/catkin_ws/src/final_olivia
```
For your environment,
```
python3 path-to-the-python script --network path-to-the-network --dataset cornell --dataset-path path-to-the-dataset --outdir path-to-the-save folder
```


