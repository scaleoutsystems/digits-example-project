[This file should be rewritten to focus on descibing the data format and what members need to do to hook up their own loca data sources]

# A federated model alliance for MNIST classification, with a CNN base model
This example illustrates how to set up an alliance and collaborate on training a basic CNN model for classifying hand-written digits (MNIST). 

First, ensure that you have access to an instance of the Scaleout Federated Platform, see: https://github.com/scaleoutsystems/federate

The guide below assumes that you have federated running on localhost:9001, and that you start Docker containers from that same host. 

Also ensure that you have installed the Scaleout Platform CLI client: https://github.com/scaleoutsystems/scaleout-cli

## Example of a project.yaml file
```yaml
    Alliance:
        id: 
        name: 
        uid: 
    Config:
        hosts:
            minio_host: localhost
            minio_port: 9000
            registrar_host: localhost
            registrar_port: 9001
    Member:
        id: 
        name: 
        state: 
    Project:
        created_by: 
        entry_points:
            predict:
                command: python3 predict.py
            train:
                command: python3 train.py
            validate:
                command: python3 validate.py
        id: 
        name: 
``` 
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

## 2. Set up an alliance member
The members of an alliance generate incremental updates to the global model upon request, by training the model on locally available data. They may also act as validators, by scoring the federated model of local test data. 

## 2.1 Download the MNIST dataset to make it available to a local trainer
In this example we are going to assume that a local member will run inside a container, and that it can access data from a volume mounted to that container. Download the complete MNIST dataset and place the folder somewhere on your host machine
https://www.kaggle.com/oddrationale/mnist-in-csv 

Create a docker volume

    $ docker volume create mnistdata

Then launch a container, mounting the volume and populating it with the datafiles (use 'docker cp'). Make sure to name the training  dataset 'train.csv',and the test dataset 'test.csv'.

```bash
$ docker run -d --name mnist_cnn_data --mount source=mnistdata,target=/app nginx:latest

$ docker cp mnist-in-csv/. mnist_cnn_data:app/ 
```

### 2.2 Build Docker images for local trainers / validators
The folder 'member' implements the local components of the federated training process.  This is the code that needs to be staged and run at each member's local site. 

Copy __project.yaml__ file from the 'model_initiator' directory and paste it into the 'member' folder. We assume that you are running the container for an existing member in the alliance. Make sure that **id**, **name** and **state** in __project.yaml__ are relevant to the member you want to run in the container. If you intend to create a new alliance member within the container, leave the member information blank. Finally, run the following:

     $ docker-compose build 

## 3. Attach members to the alliance
To launch the member in a new docker container and make it listen for contribution requests:

    $ docker run --network host --mount source=mnistdata,target=/app/data/ -it scaleout/member:latest /bin/bash
    
Note that the '--network host' flag should only be used if your federated registrar is running on localhost on the docker host.

To  create a new member with a random name and attach it to the alliance (from the container)

    $ python3 attach_to_alliance.py
    
To make the member listen to training and validation requests

    $ scaleout run client
    
Repeat the process, adding at least one more member.
    
## 4. Start federated learning  
Copy __project.yaml__ file from the 'model_initiator' directory and paste it into the 'orchestrator' folder. In this example, we assume that there is a globally available test set. To simulate that, in the 'orchestator' folder, navigate to the folder 'data' and paste test.csv there.  Then, to start training:

      $ python3 orchestrate.py <training_rounds> <number_of_trainers>
