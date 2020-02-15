
# A federated model alliance for classification of hand-written digits, using a simple Convolutional Neural Network (CNN)

The objective of this example project is to illustrates how to set up a basic federated CNN model. Here, the classical benchmark problem of classifying hand-written digits in the MNIST dataset is considered. 

## 1. Instructions for alliance members
The members of an alliance contribute with incremental updates to the global model upon request, by training on locally available data. Members will also act as validators, by scoring the global federated model on their own local training data in order to compute average/max/min training errors. 

### 1.1 Integrating your local data sources (members)

This example assumes that the member container can read training examples from a file called "train.csv", stored on a Docker  volume. When starting the member services, you will be asked to provide the ID of this volume. 

To prepare the Docker volume: 

Download the complete MNIST dataset and place the folder somewhere on your host machine https://www.kaggle.com/oddrationale/mnist-in-csv 

Create a docker volume called 'mnistdata'

    $ docker volume create mnistdata

Then launch a container (any base container will do), mounting the volume and populating it with the datafiles (use 'docker cp'). Make sure to name the training  dataset 'train.csv',and the test dataset 'test.csv'.

```bash
$ docker run -d --name mnist_cnn_data --mount source=mnistdata,target=/app nginx:latest

$ docker cp mnist-in-csv/mnist_train.csv mnist_cnn_data:app/ 
```

You can now delete your docker container if you like. 

### 1.2 Starting member trainer and validator services to join the alliance

[TODO: How do they specify the Docker volume mount?]

Download the Docker-compose file "member.yaml", then:

     $ docker-compose up -f member.yaml 
     
    
# 3. Instuctions for Alliance Admins 
TODO [Should maybe be elsewhere?]
