 # Cloud Native Vertical Federated Learning on AWS
This project introduces a reference architecture and implementation of Vertical Federated Learning (VFL) on AWS. VFL is a flavor of Federated Learning (FL) which is a distributed machine learning (ML) technique. 

Federated Learning (FL) addresses typical machine learning challenges like below:

1. Privacy control  
    Machine learning needs data for training. However, privacy control policies such as GDPR and CCPA may prevent data from moving to other location where trainig task runs.  

1. Cost of data transfer  
    Organization has their own large data lake. Machine learning often needs data across their data lakes, however as data size grows, it becomes too heavy to lift those data since it requires high bandwidth and stable network connections.

FL doesn't require data being centralized, it doesn't disclose data to other parties while building the model.

There are two flavors of FL which cover different use cases, Horizontal Federated Learning (HFL) and Vertical Federated Learning (VFL). This project focuses on VFL.

## Vertical Federated Learning (VFL)
VFL consists of the server and multiple clients, which work together to train a global ML model. They exchange intermediate data (i.e., embeddings and gradients) to train the model while the server and the clients don't access their local data each other. Each client doesn't need to have the same features each other, which is the flexible part of VFL.

![VFL architecture](_static/img/VFL%20architecture.png)

## VFL architecture on AWS
This is a reference implementation of VFL on AWS. The workflow on the server is managed by [AWS Step Functions](https://aws.amazon.com/step-functions/) state machine which orchestrates [AWS Lamda](https://aws.amazon.com/lambda/) functions and the steps of interaction with all clients. The server and each client communicate through Amazon SQS messages. Amazon S3 bucket works as an intermediary between the server and client for exchanging the object files required for building a model.  

![VFL on AWS](_static/img/VFL%20on%20AWS.png)

### AWS Step Functions state machines
The training workflow consists of three state machines, main, training and validation. The main state machine triggers a training and validation state machine by calling AWS Step Functions *StartExecution* API.  

[Callback Pattern](https://docs.aws.amazon.com/step-functions/latest/dg/callback-task-sample-sqs.html) is used for integrating the client's training and validation steps with the state machine.

![State machines](_static/img/State%20machines.png)

## Experiment
Our experiment with [Adult DataSet](https://archive.ics.uci.edu/ml/datasets/Adult) in UCI machine learning repository [[1](#uci-ml-repo)] shows the effectiveness of VFL. In the experiment, the number of clients varies between 2 and 4 and the clients are deployed in different AWS regions as shown below:

|Client|Region|
| :--- | :--- |
|#1|us-west-2|
|#2|us-east-1|
|#3|eu-west-1|
|#4|ap-northeast-1|

The following table describes the ROC-AUC and total training time of this experiment under the number of epoch = 10 and the batch size = 1,024.
It describes the acurracy of the model is improved as the number of clients increases. Note that the server and the clients don't access their local data each other while training.

|Clients|ROC-AUC|Training Time [s]|
| :--- | :--- | :--- |
| #1 + #2 | 0.8117 | 1,187 |
| #1 + #2 + #3 | 0.8887 | 1,575 |
| #1 + #2 + #3 + #4 | 0.9007 | 1,758 |

## Get Started
You can start an experiment with setting up a server and clients.

### Sample data preparation
Before setting up server and clients, the sample data needs to be prepared for the experiment.
**Python 3.8** is required to prepare the data.

1. Clone the repository
    ```shell
    git clone https://github.com/docomoinnovations/AWS-Serverless-Vertical-Federated-Learning
    cd AWS-Serverless-Vertical-Federated-Learning
    ```

1. Install libraries
    Run `pip` to install the required libraries.
    ```shell
    pip install -r requirements.txt
    ```

1. Run the command below to prepare the data.
    ```shell
    python init_data.py
    ```

For this experiment, [Adult DataSet](https://archive.ics.uci.edu/ml/datasets/Adult) in UCI machine learning repository [[1](#uci-ml-repo)] is used.  The dataset consists of 14 features. For the VFL simulation, the features are divided into four subsets of features. Each subset of features is associated with a client while the server has only the label data.

### Server and Client setup
The server and clients can be set up by following the instructions below:
1. [Server Setup](server/README.md)
1. [Client Setup](client/README.md)

### Run experiment
Both the server and the clients are set up, you can run the experiment.

1. Run the server (Replace [YourStackName] to the name of the stack deployed at [the stack deployment step](server/README.md#deploying-the-server))
    ```shell
    STACK_NAME=[YourStackName] && STATE_MACHINE_ARN=$(aws cloudformation describe-stacks --stack-name "${STACK_NAME}" --query 'Stacks[0].Outputs[?OutputKey==`StateMachineArn`].OutputValue' --output text) && aws stepfunctions start-execution --state-machine-arn ${STATE_MACHINE_ARN}
    ```

2. Run the clients  
    Now, you can run the client with the command below:
    ```shell
    python local_training.py [client-number]
    ```

    For example, run as below if you want to run the VFL client #1.
    ```shell
    python local_training.py 1
    ```

    You need to run the same number clients as the parameter *NumberOfClients* you put on [Deploying the template](server/README.md#deploying-the-server). If you put *2* as the parameter, it means VFL consists of 2 clients and the server. So, you have to run the client #1 and #2 with `python local_training.py 1` and `python local_training.py 2`.    

    Each client communicates with the server through a queue, as shown in the table below:

    |Clinet|Queue name|Region|
    | --- | --- | --- |
    |#1|vfl-us-west-2|us-west-2|
    |#2|vfl-us-east-1|us-east-1|
    |#3|vfl-eu-west-1|eu-west-1|
    |#4|vfl-ap-northeast-1|ap-northeast-1|

### Testing model
You can evaluate the model's accuracy once the training is completed. 

[Testing Model](test/README.md)

# Dataset Reference
1. <span id="uci-ml-repo">Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.</span>