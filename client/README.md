# Client setup
The VFL client is a local Python program which can run on any location. In our example code, Amazon SQS queues, which are used for the communication between the server and the clients, are deployed in different regions since each VFL client is deployed in different regions, as shown below:

|Client|Region|
| --- | --- |
|#1|us-west-2|
|#2|us-east-1|
|#3|eu-west-1|
|#4|ap-northeast-1|

As the distance between the client and the queue becomes shorter, the latency of the communication between the server and the client becomes less. Even though we assume the clients are also deployed on the location close to each region, you can run them on any location such as your laptop.

The tools below are necessary for setting up the client.

- Python 3.8

## Create IAM user for clients
VFL client program needs IAM credentials to access AWS services. The program accesses Amazon SQS, Amazon S3 and AWS Step Functions APIs. The IAM policy and group are created when [deploying the server](../server/README.md#deploying-the-server).

You need to create IAM user and its access key ID and secret access key.\
It is recommended to create an IAM user for each VFL client. This will prevent the other clients from accessing the intermediate files.

1. [Creating an IAM user in your AWS account](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html)
1. [Managing access keys for IAM users](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html)

Then, add the users to the group to grant access to AWS services.
- [Add users to group](https://docs.aws.amazon.com/singlesignon/latest/userguide/adduserstogroups.html)

The group name created by [deploying the server](../server/README.md#deploying-the-server) can be found by the following command:

```shell
STACK_NAME=[STACK_NAME]; aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].Outputs[?OutputKey==`IAMGroupNameForClient`].OutputValue' --output text
```
*STACK_NAME* is a name of the CloudFormation stack. It should be `[STACK_NAME]-IAMGroupForClient-XXXXXXXX`.


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

## Configure credentials
Configure IAM credentials on each VFL client.
If you configure the credentials with environmet variables, set environment variables bellow:
- [Configure credentials with environment variables (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html#environment-variables)
