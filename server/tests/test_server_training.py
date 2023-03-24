import pytest
import boto3
import random
import string
import tempfile
import torch
import numpy as np
from functions.server_training.server_training import (
    S3Url,
    ShuffledIndex,
    ServerTrainer,
    TrainingSession,
    DataSet,
)
from functions.init_server.serverinit import set_shuffled_index


@pytest.mark.parametrize(
    ("s3_url", "bucket", "key", "file_name"),
    [
        ("s3://test/example.jpg", "test", "example.jpg", "example.jpg"),
        (
            "s3://test/prefix/shuffled-index.npy",
            "test",
            "prefix/shuffled-index.npy",
            "shuffled-index.npy",
        ),
        (
            "s3://test.example.com/prefix/shuffled-index.npy",
            "test.example.com",
            "prefix/shuffled-index.npy",
            "shuffled-index.npy",
        ),
    ],
)
def test_s3_url(s3_url, bucket, key, file_name):
    s3_url_obj = S3Url(s3_url)
    assert s3_url_obj.url == s3_url
    assert s3_url_obj.bucket == bucket
    assert s3_url_obj.key == key
    assert s3_url_obj.file_name == file_name


def test_init_dataset():
    dataset_dir = "functions/server_training"
    dataset = DataSet(
        label=f"{dataset_dir}/tr_y.npy",
        uid=f"{dataset_dir}/tr_uid.npy",
        va_label=f"{dataset_dir}/va_y.npy",
        va_uid=f"{dataset_dir}/va_uid.npy",
    )

    assert (
        dataset.label.dtype
        == torch.FloatTensor(
            np.load(f"{dataset_dir}/tr_y.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.label) == len(
        torch.FloatTensor(np.load(f"{dataset_dir}/tr_y.npy", allow_pickle=False))
    )
    assert (
        dataset.uid.dtype
        == torch.LongTensor(
            np.load(f"{dataset_dir}/tr_uid.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.uid) == len(
        torch.LongTensor(np.load(f"{dataset_dir}/tr_uid.npy", allow_pickle=False))
    )
    assert (
        dataset.va_label.dtype
        == torch.FloatTensor(
            np.load(f"{dataset_dir}/va_y.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.va_label) == len(
        torch.FloatTensor(np.load(f"{dataset_dir}/va_y.npy", allow_pickle=False))
    )
    assert (
        dataset.va_uid.dtype
        == torch.LongTensor(
            np.load(f"{dataset_dir}/va_uid.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.va_uid) == len(
        torch.LongTensor(np.load(f"{dataset_dir}/va_uid.npy", allow_pickle=False))
    )


@pytest.fixture
def shuffled_index_url() -> S3Url:
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

    shuffled_index_url = set_shuffled_index(
        "functions/init_server/tr_uid.npy",
        "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
        bucket.name,
        prefix="common/",
    )

    yield S3Url(shuffled_index_url)

    bucket.objects.all().delete()
    bucket.delete()


def test_shuffled_index(shuffled_index_url: S3Url):
    shuffled_index = ShuffledIndex(shuffled_index_url)
    bucket = shuffled_index_url.bucket
    key = shuffled_index_url.key
    file_name = shuffled_index_url.file_name
    shuffled_index_obj = boto3.resource("s3").Object(bucket, key)
    with tempfile.TemporaryDirectory() as tmpdirname:
        shuffled_index_obj.download_file(f"{tmpdirname}/{file_name}")
        donwloaded_shuffled_index = torch.LongTensor(
            np.load(f"{tmpdirname}/{file_name}", allow_pickle=False)
        )
        assert shuffled_index.s3_url == shuffled_index_url
        assert shuffled_index.index.tolist() == donwloaded_shuffled_index.tolist()


@pytest.fixture
def bucket_name() -> str:
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

    set_shuffled_index(
        "functions/init_server/tr_uid.npy",
        "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
        bucket.name,
        prefix="common/",
    )

    yield bucket.name

    bucket.objects.all().delete()
    bucket.delete()


@pytest.fixture
def shuffled_index() -> ShuffledIndex:
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
    s3_url = set_shuffled_index(
        "functions/init_server/tr_uid.npy",
        "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
        bucket.name,
        prefix="common/",
    )

    yield ShuffledIndex(S3Url(s3_url))

    bucket.objects.all().delete()
    bucket.delete()


@pytest.mark.parametrize(
    (
        "task_name",
        "num_of_clients",
        "epoch_index",
        "batch_index",
        "batch_size",
        "va_batch_index",
    ),
    [
        (
            "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
            4,
            0,
            0,
            1024,
            0,
        )
    ],
)
def test_training_session(
    task_name,
    num_of_clients,
    epoch_index,
    batch_index,
    batch_size,
    va_batch_index,
    shuffled_index,
):
    session = TrainingSession(
        task_name=task_name,
        num_of_clients=num_of_clients,
        epoch_index=epoch_index,
        batch_index=batch_index,
        batch_size=batch_size,
        va_batch_index=va_batch_index,
        shuffled_index=shuffled_index,
    )
    assert session.task_name == task_name
    assert session.num_of_clients == num_of_clients
    assert session.epoch_index == epoch_index
    assert session.batch_index == batch_index
    assert session.batch_size == batch_size
    assert session.va_batch_index == va_batch_index
    assert session.shuffled_index.tolist() == shuffled_index.index.tolist()


@pytest.mark.parametrize(
    (
        "task_name",
        "num_of_clients",
        "epoch_index",
        "batch_index",
        "batch_size",
        "va_batch_index",
    ),
    [
        (
            "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
            4,
            0,
            0,
            1024,
            0,
        )
    ],
)
def test_init_server_trainer(
    task_name,
    num_of_clients,
    epoch_index,
    batch_index,
    batch_size,
    va_batch_index,
    shuffled_index,
):
    dataset_dir = "functions/server_training"
    label = f"{dataset_dir}/tr_y.npy"
    uid = f"{dataset_dir}/tr_uid.npy"
    va_label = f"{dataset_dir}/va_y.npy"
    va_uid = f"{dataset_dir}/va_uid.npy"

    dataset = DataSet(label=label, uid=uid, va_label=va_label, va_uid=va_uid)
    training_session = TrainingSession(
        task_name=task_name,
        num_of_clients=num_of_clients,
        epoch_index=epoch_index,
        batch_index=batch_index,
        batch_size=batch_size,
        va_batch_index=va_batch_index,
        shuffled_index=shuffled_index,
    )

    server_trainer = ServerTrainer(
        training_session=training_session, s3_bucket=bucket_name, dataset=dataset
    )
    assert server_trainer.task_name == training_session.task_name
    assert server_trainer.num_of_clients == training_session.num_of_clients
    assert server_trainer.client_ids == [
        str(i + 1) for i in range(training_session.num_of_clients)
    ]
    assert server_trainer.epoch_index == training_session.epoch_index
    assert server_trainer.batch_index == training_session.batch_index
    assert server_trainer.batch_size == training_session.batch_size
    assert server_trainer.va_batch_index == training_session.va_batch_index
    assert server_trainer.s3_bucket == bucket_name
    assert len(server_trainer.tr_y) == len(dataset.label)
    assert len(server_trainer.tr_uid) == len(dataset.uid)
    assert len(server_trainer.va_y) == len(dataset.va_label)
    assert len(server_trainer.va_uid) == len(dataset.va_uid)
    assert (
        server_trainer.shuffled_index.tolist()
        == training_session.shuffled_index.tolist()
    )
