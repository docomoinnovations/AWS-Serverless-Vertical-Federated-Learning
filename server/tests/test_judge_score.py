import pytest
import boto3
import random
import string
import tempfile
import json
from functions.judge_score.judge_score import (
    get_score,
    save_score,
    Score,
    lambda_handler,
)


@pytest.fixture
def bucket_name() -> str:
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

    yield bucket.name

    bucket.objects.all().delete()
    bucket.delete()


@pytest.mark.parametrize(
    ("best_score", "patience_counter"), [(0.8110, 2), (-1, 0), (1, 100)]
)
def test_get_score(best_score, patience_counter, bucket_name):
    task_name = "VFL-TAKS-YYYY-MM-DD-HH-mm-ss"
    bucket = boto3.resource("s3").Bucket(bucket_name)
    key = f"server/{task_name}-score.json"
    with tempfile.TemporaryDirectory() as tmpdirname:
        score_data = {
            "best_score": best_score,
            "patience_counter": patience_counter,
        }

        file_name = f"{tmpdirname}/{task_name}-score.json"
        with open(file_name, "w") as f:
            json.dump(score_data, f)

        bucket.upload_file(file_name, key)

    score = get_score(s3_bucket=bucket_name, key=key)
    assert score.best_score == best_score
    assert score.patience_counter == patience_counter


@pytest.mark.parametrize(
    ("best_score", "patience_counter"), [(0.8110, 2), (-1, 0), (1, 100)]
)
def test_save_score(best_score, patience_counter, bucket_name):
    task_name = "VFL-TAKS-YYYY-MM-DD-HH-mm-ss"
    key = f"server/{task_name}-score.json"
    score = Score(best_score=best_score, patience_counter=patience_counter)
    save_score(score=score, s3_bucket=bucket_name, key=key)
    bucket = boto3.resource("s3").Bucket(bucket_name)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_name = f"{task_name}-score.json"
        local_file_path = f"{tmpdirname}/{file_name}"
        bucket.download_file(key, local_file_path)
        with open(local_file_path, "r") as f:
            score_data = json.load(f)
            assert score_data["best_score"] == best_score
            assert score_data["patience_counter"] == patience_counter


@pytest.fixture
def events(bucket_name):
    vfl_bucket = bucket_name
    return [
        [
            {
                "TaskName": "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
                "BatchIndex": 10,
                "BatchSize": 1024,
                "VaBatchIndex": 0,
                "BatchCount": 30,
                "VaBatchCount": 3,
                "ShuffledIndexPath": "s3://vfl-test/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.pt",
                "EpochIndex": 0,
                "EpochCount": 5,
                "IsNextBatch": "true",
                "IsNextVaBatch": "true",
                "IsNextEpoch": "true",
                "VaAuc": 0.7500,
                "Patience": 3,
                "SparseEncoding": True,
                "SparseLambda": 0.1,
                "SqsUrl": "https://sqs.us-west-2.amazonaws.com/123456789012/vfl-queue-1",
                "VFLBucket": vfl_bucket,
            },
        ],
        [
            {
                "TaskName": "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
                "BatchIndex": 10,
                "BatchSize": 5000,
                "VaBatchIndex": 0,
                "BatchCount": 30,
                "VaBatchCount": 3,
                "ShuffledIndexPath": "s3://vfl-test/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.pt",
                "EpochIndex": 3,
                "EpochCount": 10,
                "IsNextBatch": "true",
                "IsNextVaBatch": "true",
                "IsNextEpoch": "true",
                "VaAuc": 0.7500,
                "Patience": 3,
                "SqsUrl": "https://sqs.us-west-2.amazonaws.com/123456789012/vfl-queue-1",
                "SparseEncoding": False,
                "SparseLambda": 0.5,
                "VFLBucket": vfl_bucket,
            },
            {
                "TaskName": "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
                "BatchIndex": 10,
                "BatchSize": 5000,
                "VaBatchIndex": 0,
                "BatchCount": 30,
                "VaBatchCount": 3,
                "ShuffledIndexPath": "s3://vfl-test/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.pt",
                "EpochIndex": 3,
                "EpochCount": 10,
                "IsNextBatch": "true",
                "IsNextVaBatch": "true",
                "IsNextEpoch": "true",
                "VaAuc": 0.7500,
                "Patience": 3,
                "SqsUrl": "https://sqs.us-west-2.amazonaws.com/123456789012/vfl-queue-2",
                "SparseEncoding": False,
                "SparseLambda": 0.5,
                "VFLBucket": vfl_bucket,
            },
            {
                "TaskName": "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
                "BatchIndex": 10,
                "BatchSize": 5000,
                "VaBatchIndex": 0,
                "BatchCount": 30,
                "VaBatchCount": 3,
                "ShuffledIndexPath": "s3://vfl-test/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.pt",
                "EpochIndex": 3,
                "EpochCount": 10,
                "IsNextBatch": "true",
                "IsNextVaBatch": "true",
                "IsNextEpoch": "true",
                "Patience": 3,
                "VaAuc": 0.7500,
                "SqsUrl": "https://sqs.us-west-2.amazonaws.com/123456789012/vfl-queue-3",
                "SparseEncoding": False,
                "SparseLambda": 0.5,
                "VFLBucket": vfl_bucket,
            },
        ],
    ]


def test_lambda_handler(events):
    expected_keys = [
        "TaskName",
        "BatchIndex",
        "BatchCount",
        "BatchSize",
        "VaBatchIndex",
        "VaBatchCount",
        "IsNextBatch",
        "IsNextVaBatch",
        "EpochIndex",
        "IsNextEpoch",
        "Patience",
        "ShuffledIndexPath",
        "SparseEncoding",
        "SparseLambda",
        "IsBestScore",
        "SqsUrl",
        "VFLBucket",
    ]
    for event in events:
        res = lambda_handler(event, {})
        assert len(res) == len(event)
        for key in expected_keys:
            for client in res:
                assert key in client
