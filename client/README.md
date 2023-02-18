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

## AWS IAM permissions
Client needs the following IAM permissions to access AWS APIs.
- `sqs:GetQueueUrl`
- `sqs:ReceiveMessage`
- `sqs:DeleteMessage`
- `s3:GetObject`
- `s3:PutObject`
- `state:SendTaskSuccess`

This is the example policy for client.

```json
{
    "Version": "2012-10-17",
    "Statement": {
        "Effect": "Allow",
        "Action": [
            "sqs:GetQueueUrl",
            "sqs:ReceiveMessage",
            "sqs:DeleteMessage",
            "s3:GetObject",
            "s3:PutObject",
            "states:SendTaskSuccess"
        ],
        "Resource": "*"
    }
}
```


## Installing the libraries
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
