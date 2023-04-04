import pytest
import boto3
import torch
import numpy as np
import random
import string
import json
import tempfile
from functions.init_server.serverinit import (
    get_queues_to_create,
    create_sqs,
    lambda_handler,
    get_sample_count,
    get_batch_count,
    set_shuffled_index,
)


@pytest.mark.parametrize(
    ("num_of_queues", "expected"),
    [
        (
            1,
            [
                {
                    "Name": "vfl-us-east-1",
                    "Region": "us-east-1",
                    "ClientId": "1",
                }
            ],
        ),
        (
            2,
            [
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
            ],
        ),
        (
            3,
            [
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
            ],
        ),
        (
            4,
            [
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
            ],
        ),
    ],
)
def test_get_queues_to_create(num_of_queues, expected):
    assert get_queues_to_create(num_of_queues) == expected


@pytest.fixture
def prep_create_sqs():
    sqs_region = "us-east-1"
    sqs_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )
    client_id = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(3)]
    )
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    policy = {
        "Version": "2012-10-17",
        "Id": f"PolicyForVFLClient{client_id}",
        "Statement": [
            {
                "Sid": "AllowQueueAction",
                "Effect": "Allow",
                "Principal": {
                    "AWS": f"arn:aws:iam::{account_id}:root",
                },
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
            },
        ],
    }

    yield {
        "Region": sqs_region,
        "Name": sqs_name,
        "ClientId": client_id,
        "Policy": policy,
    }

    sqs_url = boto3.client("sqs", region_name=sqs_region).get_queue_url(
        QueueName=sqs_name
    )["QueueUrl"]
    boto3.client("sqs", region_name=sqs_region).delete_queue(QueueUrl=sqs_url)


def test_create_sqs(prep_create_sqs):
    region = prep_create_sqs["Region"]
    name = prep_create_sqs["Name"]
    client_id = prep_create_sqs["ClientId"]
    expected_policy = prep_create_sqs["Policy"]

    sqs_url = create_sqs(region, name, client_id)
    client = boto3.client("sqs", region_name=region)
    expected = client.get_queue_url(QueueName=name)["QueueUrl"]
    assert sqs_url == expected

    policy = json.loads(
        client.get_queue_attributes(
            QueueUrl=sqs_url,
            AttributeNames=["Policy"],
        )[
            "Attributes"
        ]["Policy"]
    )
    assert policy == expected_policy


@pytest.mark.parametrize(
    ("dataset", "expected"),
    [
        ("functions/init_server/tr_uid.npy", 29304),
        ("functions/init_server/va_uid.npy", 3257),
    ],
)
def test_get_sample_count(dataset, expected):
    assert get_sample_count(dataset) == expected


@pytest.mark.parametrize(
    ("dataset", "batch_size", "expected"),
    [
        ("functions/init_server/tr_uid.npy", 1024, 29),
        ("functions/init_server/va_uid.npy", 1024, 4),
        ("functions/init_server/tr_uid.npy", 4096, 8),
        ("functions/init_server/va_uid.npy", 4096, 1),
    ],
)
def test_get_batch_count(dataset, batch_size, expected):
    assert get_batch_count(dataset, batch_size) == expected


@pytest.fixture
def bucket() -> str:
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

    yield bucket.name

    bucket.objects.all().delete()
    bucket.delete()


def test_set_shuffled_index(bucket):
    task_name = "VFL-TAKS-YYYY-MM-DD-HH-mm-ss"
    dataset = "functions/init_server/tr_uid.npy"

    prefix = "common/"
    key = f"{prefix}{task_name}-shuffled-index.npy"
    s3_url = set_shuffled_index(
        dataset=dataset, task_name=task_name, bucket=bucket, prefix=prefix
    )
    assert s3_url == f"s3://{bucket}/{key}"

    shuffled_index_obj = boto3.resource("s3").Object(bucket, key)
    with tempfile.TemporaryDirectory() as tmpdirname:
        shuffled_index_obj.download_file(f"{tmpdirname}/shuffled-index.npy")
        shuffled_index = torch.LongTensor(
            np.load(f"{tmpdirname}/shuffled-index.npy", allow_pickle=False)
        )
        assert get_sample_count(dataset) == len(shuffled_index)


@pytest.fixture
def event():
    num_of_clients = 4
    s3_bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )
    s3_bucket = boto3.resource("s3").Bucket(s3_bucket_name)
    s3_bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    batch_size = 1024
    epoch_count = 1000

    yield {
        "num_of_clients": num_of_clients,
        "s3_bucket": s3_bucket_name,
        "batch_size": batch_size,
        "epoch_count": epoch_count,
    }

    s3_bucket.objects.all().delete()
    s3_bucket.delete()


def test_lambda_handler(event):
    res = lambda_handler(event, {})
    assert len(res) == event["num_of_clients"]
    for client in res:
        assert "SqsUrl" in client
        assert "BatchIndex" in client
        assert "BatchCount" in client
        assert "VaBatchIndex" in client
        assert "VaBatchCount" in client
        assert "EpochIndex" in client
        assert "IsNextEpoch" in client
        assert "IsNextBatch" in client
        assert "IsNextVaBatch" in client
        assert "TaskName" in client
        assert "ShuffledIndexPath" in client
