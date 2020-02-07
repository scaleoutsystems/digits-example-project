[This file should be rewritten to focus on descibing the data format and what members need to do to hook up their own loca data sources]

# A federated model alliance for classification of hand-written digits, using a simple Convolutional Neural Network (CNN)

The objective of this alliance model is to illustrates how to set up an alliance and collaborate on training a basic CNN model using the classical benchmark problem of classifying hand-written digits in the MNIST dataset. 

## 1. Set up an alliance model
Navigate to model_initiator folder where you will find 4 files in total. You need to have a configuration file for your project (project.yaml)

Either create one or download it from our UI client.

To set up an alliance model, you need to go through the following steps:
1. Initialize an alliance
2. Add a member to the alliance
3. Let the member create a project
4. Initialize a model in the newly created project
```bash
python3 alliance_initiator.py

python3 project_initiator.py

python3 init_alliance_model.py
```
  
This will invoke Keras, and create a base CNN model. This model will serve as a starting point for federated training. Make sure to record the __uid__ under __Alliance__ in project.yaml, this will be needed to attach members to the alliance. 

## 2. Instructions for alliance members
The members of an alliance generate incremental updates to the global model upon request, by training the model on locally available data. They will also act as validators, by scoring the federated model of local training data to compute average/max/min training errors. 

## 2.1 Integrating your local data sources

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

### 2.2 Start member trainer and validator services to join the alliance

[TODO: How do they specify the Docker volume mount?]

Download the Docker-compose file "member.yaml", then:

     $ docker-compose up -f member.yaml 
     
    
## 3. Start federated learning  
TODO
