import math
import torch
import boto3
import pandas
import numpy as np
import os
import json
import tempfile
from time import gmtime, strftime


def get_queues_to_create(num_of_queues):
    if num_of_queues < 1 or num_of_queues > 4:
        print("num_of_clients must be 1 to 4")
        return []

    queues = [
        {
            "Name": "vfl-us-east-1",
            "Region": "us-east-1",
            "ClientId": "1",
        },
        {
            "Name": "vfl-us-west-2",
            "Region": "us-west-2",
            "ClientId": "2",
        },
        {
            "Name": "vfl-eu-west-1",
            "Region": "eu-west-1",
            "ClientId": "3",
        },
        {
            "Name": "vfl-ap-northeast-1",
            "Region": "ap-northeast-1",
            "ClientId": "4",
        },
    ]

    return queues[:num_of_queues]


def create_sqs(sqs_region, sqs_name, client_id):
    client = boto3.client("sqs", region_name=sqs_region)
    client.create_queue(QueueName=sqs_name)

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    policy = {
        "Version": "2012-10-17",
        "Id": f"PolicyForVFLClient{client_id}",
        "Statement": [
            {
                "Sid": "AllowQueueAction",
                "Effect": "Allow",
                "Principal": {"AWS": f"arn:aws:iam::{account_id}:root"},
                "Action": [
                    "sqs:GetQueueUrl",
                    "sqs:ReceiveMessage",
                    "sqs:DeleteMessage",
                ],
                "Resource": f"arn:aws:sqs:{sqs_region}:{account_id}:{sqs_name}",
                "Condition": {
                    "StringEquals": {
                        "aws:PrincipalTag/vfl-client-id": client_id,
                    }
                },
            }
        ],
    }

    queue_url = client.get_queue_url(QueueName=sqs_name)["QueueUrl"]
    client.set_queue_attributes(
        QueueUrl=queue_url, Attributes={"Policy": json.dumps(policy)}
    )
    return queue_url


# delete all message in a sqs queue
def sqs_purge(sqs_region, sqs_name):
    sqs_client = boto3.client("sqs", region_name=sqs_region)
    queue_url = sqs_client.get_queue_url(QueueName=sqs_name)["QueueUrl"]

    response = sqs_client.purge_queue(QueueUrl=queue_url)
    return response


# Get number of sample data
def get_sample_count(file):
    uid = torch.LongTensor(np.load(file, allow_pickle=False))
    return len(uid)


# Get batch count number
def get_batch_count(dataset, batch_size):
    tr_sample_count = get_sample_count(dataset)
    return math.ceil(tr_sample_count / batch_size)


# Shuffled index
def set_shuffled_index(dataset, task_name, bucket, prefix=""):
    sample_count = get_sample_count(dataset)
    shuffled_index_file_name = f"{task_name}-shuffled-index.npy"
    key = f"{prefix}{shuffled_index_file_name}"
    shuffled_index = torch.randperm(sample_count)
    with tempfile.TemporaryDirectory() as tmpdirname:
        np.save(
            f"{tmpdirname}/{shuffled_index_file_name}",
            shuffled_index.numpy(),
            allow_pickle=False,
        )
        s3 = boto3.resource("s3")
        s3.meta.client.upload_file(
            f"/{tmpdirname}/{shuffled_index_file_name}",
            bucket,
            key,
        )
    return f"s3://{bucket}/{key}"


def lambda_handler(event, context):
    print(event)
    dir = os.path.dirname(os.path.abspath(__file__))

    num_of_clients = int(event["num_of_clients"])
    queues_to_create = get_queues_to_create(num_of_clients)
    queues = []

    for queue in queues_to_create:
        region = queue["Region"]
        name = queue["Name"]
        client_id = queue["ClientId"]
        queues.append(create_sqs(sqs_region=region, sqs_name=name, client_id=client_id))
        sqs_purge(region, name)

    # S3 bucket to store shuffled index
    s3_bucket = event["s3_bucket"]

    task_name = "VFL-Task-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())
    batch_size = int(event["batch_size"])
    batch_count = get_batch_count(dataset=f"{dir}/tr_uid.npy", batch_size=batch_size)
    va_batch_count = get_batch_count(dataset=f"{dir}/va_uid.npy", batch_size=batch_size)
    shuffled_index_path = set_shuffled_index(
        dataset=f"{dir}/tr_uid.npy",
        task_name=task_name,
        bucket=s3_bucket,
        prefix="common/",
    )
    epoch_count = int(event["epoch_count"])

    response = []
    for queue in queues:
        response.append(
            {
                "SqsUrl": queue,
                "BatchIndex": 0,
                "BatchCount": batch_count,
                "VaBatchIndex": 0,
                "VaBatchCount": va_batch_count,
                "EpochIndex": 0,
                "IsNextEpoch": 0 + 1 < epoch_count,
                "IsNextBatch": 0 + 1 < batch_count,
                "IsNextVaBatch": 0 + 1 < va_batch_count,
                "TaskName": task_name,
                "ShuffledIndexPath": shuffled_index_path,
            }
        )

    return response
