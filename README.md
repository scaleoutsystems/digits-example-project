
# A federated model alliance for classification of hand-written digits, using a simple Convolutional Neural Network (CNN)

The objective of this example project is to illustrates how to set up a basic federated CNN model. Here, the classical benchmark problem of classifying hand-written digits in the MNIST dataset is considered. 

## 1. Instructions for alliance members
The members of an alliance contribute with incremental updates to the global model upon request, by training on locally available data. Members will also act as validators, by scoring the global federated model on their own local training data in order to compute average/max/min training errors. 

### 1.1 Integrating your local data sources (members)

This example assumes that the member places training examples in a CSV file "train.csv" in the folder "data" (see project/read_data.py). Download the complete MNIST dataset from https://www.kaggle.com/oddrationale/mnist-in-csv and place traing data in said folder (remember to rename it to 'train.csv') 

### 1.2 Starting member trainer and validator services to join the alliance

[TODO: How do they join the alliance.]

Download the Docker-compose file "member.yaml", then:

     $ docker-compose up -f member.yaml 
     
    
## 2. Instuctions for Alliance Admins 
TODO [Should maybe be elsewhere?]
