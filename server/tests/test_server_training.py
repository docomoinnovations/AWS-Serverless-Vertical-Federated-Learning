import pytest
import boto3
import random
import string
import torch
from functions.server_training.server_training import (
    ServerTrainer,
    TrainingSession,
    DataSet,
)
from functions.init_server.serverinit import set_shuffled_index


def test_init_dataset():
    dataset_dir = "functions/server_training"
    dataset = DataSet(
        label=f"{dataset_dir}/tr_y.pt",
        uid=f"{dataset_dir}/tr_uid.pt",
        va_label=f"{dataset_dir}/va_y.pt",
        va_uid=f"{dataset_dir}/va_uid.pt",
    )
    assert len(dataset.label) == len(torch.load(f"{dataset_dir}/tr_y.pt"))
    assert len(dataset.uid) == len(torch.load(f"{dataset_dir}/tr_uid.pt"))
    assert len(dataset.va_label) == len(torch.load(f"{dataset_dir}/va_y.pt"))
    assert len(dataset.va_uid) == len(torch.load(f"{dataset_dir}/va_uid.pt"))


@pytest.fixture
def bucket_name() -> str:
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})

    set_shuffled_index(
        f"functions/init_server/tr_uid.pt", "VFL-TAKS-YYYY-MM-DD-HH-mm-ss", bucket.name
    )

    yield bucket.name

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
):
    session = TrainingSession(
        task_name=task_name,
        num_of_clients=num_of_clients,
        epoch_index=epoch_index,
        batch_index=batch_index,
        batch_size=batch_size,
        va_batch_index=va_batch_index,
    )
    assert session.task_name == task_name
    assert session.num_of_clients == num_of_clients
    assert session.epoch_index == epoch_index
    assert session.batch_index == batch_index
    assert session.batch_size == batch_size
    assert session.va_batch_index == va_batch_index


@pytest.mark.parametrize(
    ("training_session"),
    [
        TrainingSession(
            task_name="VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
            num_of_clients=4,
            epoch_index=0,
            batch_index=0,
            batch_size=1024,
            va_batch_index=0,
        )
    ],
)
def test_init_server_trainer(training_session, bucket_name):
    dataset_dir = "functions/server_training"
    label = f"{dataset_dir}/tr_y.pt"
    uid = f"{dataset_dir}/tr_uid.pt"
    va_label = f"{dataset_dir}/va_y.pt"
    va_uid = f"{dataset_dir}/va_uid.pt"

    dataset = DataSet(label=label, uid=uid, va_label=va_label, va_uid=va_uid)

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
