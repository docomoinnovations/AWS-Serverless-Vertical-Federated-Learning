# Server setup
The server can be setup by [AWS Serverless Application Model (SAM)](https://aws.amazon.com/serverless/sam/).  An AWS account and the tools below are necessary for the setup.
- Python 3.8
- [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html)
- [Docker](https://docs.docker.com/)

## Preparing the sample data
The sample data preparation is required before deploying the server. If you haven't prepared it yet, please follow [Sample data preparation](../README.md#sample-data-preparation) before going to the next step.

## Installing the libraries
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

## Deploying the server
1. Building AWS SAM template  
    Before deploying the server, you need to build the template. The build process includes building the container image which is used for the AWS Lambda function. AWS SAM helps to build them by a simple command.
    ```shell
    cd server
    sam build
    ```
2. Deploying the template  
    Then, you're ready to deploy the server on your AWS account by the command below:
    ```shell
    sam deploy --guided
    ```
    You will be asked to put the following parameters. Put the value as you want and deploy it.

    |Parameters|Description|Default value|
    | --- | --- | --- |
    |Stack Name| CloudFormation stack name which is deployed by AWS SAM| sam-app |
    |AWS Region| AWS Region to deploy server | us-west-2 |
    |NumOfClients| Number of clients which the VFL consists of, which must be 2 - 4 | 4 |
    |BatchSize| The batch size for the training | 1024 |
    |EpochCount| The epoch count for the training | 10 |
    |Patience| The threshold of consective epoch count to stop the training if the model accuracy is not improved  | 3 |

    Then, you may be asked a couple of questions related to AWS SAM configurations and you may choose the preference.
    Please make sure that answer `Yes` if you're asked `Allow SAM CLI IAM role creation [Y/n]`.
