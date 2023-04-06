# Client setup
The VFL client is a local Python program which can run on any location. In our example code, Amazon SQS queues, which are used for the communication between the server and the clients, are deployed in different regions since each VFL client is deployed in different regions, as shown below:

|Client|Region|
| --- | --- |
|#1|us-east-1|
|#2|us-west-2|
|#3|eu-west-1|
|#4|ap-northeast-1|

As the distance between the client and the queue becomes shorter, the latency of the communication between the server and the client becomes less. Even though we assume the clients are also deployed on the location close to each region, you can run them on any location such as your laptop.

The tools below are necessary for setting up the client.

- Python 3.8

## Create IAM user for clients
VFL client program needs IAM credentials to access AWS services. The program accesses Amazon SQS, Amazon S3 and AWS Step Functions APIs. The IAM policy and group are created when [deploying the server](../server/README.md#deploying-the-server).

You need to create IAM user and its access key ID and secret access key.\
It is recommended to create an IAM user for each VFL client. This will prevent the other clients from accessing the intermediate files.

We provide a script to create the IAM users for clients.

1. Run the commands to download the script
    ```shell
    git clone https://github.com/docomoinnovations/AWS-Serverless-Vertical-Federated-Learning
    cd AWS-Serverless-Vertical-Federated-Learning/client/utils
    ```
1. Install the libraries
    ```shell
    pip install -r requirements.txt
    ```
1. Run the script to create users
    ```shell
    python create_iam_users.py --stack-name [STACK_NAME] --region [REGION]
    ```
    *STACK_NAME* is a name of the CloudFormation stack created when [deploying the server](../server/README.md#deploying-the-server).\
    *REGION* is a region of the CloudFormation stack.

    Then, you can get outputs as bellow:
    ```shell
    #############################################
    # Client ID: 1 (IAM user: vfl-client-xxxxx) #
    #############################################
    export AWS_ACCESS_KEY_ID="<AWS_ACCESS_KEY_ID>"
    export AWS_SECRET_ACCESS_KEY="<AWS_SECRET_ACCESS_KEY>"

    #############################################
    # Client ID: 2 (IAM user: vfl-client-yyyyy) #
    #############################################
    export AWS_ACCESS_KEY_ID=........
    .
    .
    .
    ```
    The outputs includes commands to set IAM user credentials for each client. You can just copy the commands, paste and run it in your terminal of each client to set the credentials.

## Install the libraries
1. Clone the repository
    ```shell
    git clone https://github.com/docomoinnovations/AWS-Serverless-Vertical-Federated-Learning
    cd AWS-Serverless-Vertical-Federated-Learning
    ```
1. Install the libraries
    Run `pip` to install the required libraries.
    ```shell
    cd client
    pip install -r requirements.txt
    ```

## Clean up IAM users
We provide a script to clean up IAM users. When you no longer need IAM credentials for VFL client, run the command to delete all users.\
Note that the command must run before deleting the CloudFormation stack of VFL server.
```shell
git clone https://github.com/docomoinnovations/AWS-Serverless-Vertical-Federated-Learning
cd AWS-Serverless-Vertical-Federated-Learning/client/utils
pip install -r requirements.txt
python clean_up_users.py --stack-name [STACK_NAME] --region [REGION]
```
*STACK_NAME* is a name of the CloudFormation stack created when [deploying the server](../server/README.md#deploying-the-server).\
*REGION* is a region of the CloudFormation stack.