import pytest
import torch
import numpy as np
import boto3
import tempfile
import random
import string
import json
from local_training import (
    ClientTrainer,
    Dataset,
    VFLSQS,
    TrainingSession,
    ClientModel,
    ShuffledIndex,
)


@pytest.mark.parametrize(
    ("client", "uid", "x", "cols", "va_uid", "va_x"),
    [
        (
            "1",
            "dataset/client1/tr_uid.npy",
            "dataset/client1/tr_x.npy",
            "dataset/client1/cols.npy",
            "dataset/client1/va_uid.npy",
            "dataset/client1/va_x.npy",
        ),
        (
            "2",
            "dataset/client2/tr_uid.npy",
            "dataset/client2/tr_x.npy",
            "dataset/client2/cols.npy",
            "dataset/client2/va_uid.npy",
            "dataset/client2/va_x.npy",
        ),
        (
            "3",
            "dataset/client3/tr_uid.npy",
            "dataset/client3/tr_x.npy",
            "dataset/client3/cols.npy",
            "dataset/client3/va_uid.npy",
            "dataset/client3/va_x.npy",
        ),
        (
            "4",
            "dataset/client4/tr_uid.npy",
            "dataset/client4/tr_x.npy",
            "dataset/client4/cols.npy",
            "dataset/client4/va_uid.npy",
            "dataset/client4/va_x.npy",
        ),
    ],
)
def test_dataset(client, uid, x, cols, va_uid, va_x):
    dataset = Dataset(client=client)
    assert (
        dataset.tr_uid.dtype == torch.LongTensor(np.load(uid, allow_pickle=False)).dtype
    )
    assert len(dataset.tr_uid) == len(
        torch.LongTensor(np.load(uid, allow_pickle=False))
    )
    assert dataset.tr_x.dtype == torch.FloatTensor(np.load(x, allow_pickle=False)).dtype
    assert len(dataset.tr_x) == len(torch.FloatTensor(np.load(x, allow_pickle=False)))
    assert dataset.tr_xcols == np.load(cols, allow_pickle=False).tolist()
    assert (
        dataset.va_uid.dtype
        == torch.LongTensor(np.load(va_uid, allow_pickle=False)).dtype
    )
    assert len(dataset.va_uid) == len(
        torch.LongTensor(np.load(va_uid, allow_pickle=False))
    )
    assert (
        dataset.va_x.dtype == torch.FloatTensor(np.load(va_x, allow_pickle=False)).dtype
    )
    assert len(dataset.va_x) == len(
        torch.FloatTensor(np.load(va_x, allow_pickle=False))
    )
    assert dataset.va_xcols == dataset.tr_xcols
    assert dataset.tr_sample_count == len(
        torch.LongTensor(np.load(uid, allow_pickle=False))
    )
    assert dataset.va_sample_count == len(
        torch.LongTensor(np.load(va_uid, allow_pickle=False))
    )


@pytest.mark.parametrize(
    (
        "task_name",
        "batch_size",
        "batch_index",
        "batch_count",
        "is_next_batch",
        "va_batch_index",
        "va_batch_count",
        "is_next_va_batch",
        "task_token",
        "server_region",
        "phase",
        "direction",
        "epoch_index",
        "is_next_epoch",
        "s3_bucket",
        "shuffled_index_path",
        "gradient_file_path",
    ),
    [
        (
            "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
            1024,
            10,
            30,
            True,
            0,
            3,
            True,
            "112233445566778899",
            "us-west-2",
            "Training",
            "Forward",
            5,
            True,
            "vfl-bucket-test",
            "s3://vfl-bucket-test/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.npy",
            "s3://vfl-bucket-test/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-gradient-1.npy",
        )
    ],
)
def test_training_session(
    task_name,
    batch_size,
    batch_index,
    batch_count,
    is_next_batch,
    va_batch_index,
    va_batch_count,
    is_next_va_batch,
    task_token,
    server_region,
    phase,
    direction,
    epoch_index,
    is_next_epoch,
    s3_bucket,
    shuffled_index_path,
    gradient_file_path,
):
    session = TrainingSession(
        task_name=task_name,
        batch_size=batch_size,
        batch_index=batch_index,
        batch_count=batch_count,
        is_next_batch=is_next_batch,
        va_batch_index=va_batch_index,
        va_batch_count=va_batch_count,
        is_next_va_batch=is_next_va_batch,
        task_token=task_token,
        server_region=server_region,
        phase=phase,
        direction=direction,
        epoch_index=epoch_index,
        is_next_epoch=is_next_epoch,
        s3_bucket=s3_bucket,
        shuffled_index_path=shuffled_index_path,
        gradient_file_path=gradient_file_path,
    )
    assert session.task_name == task_name
    assert session.batch_size == batch_size
    assert session.batch_index == batch_index
    assert session.batch_count == batch_count
    assert session.is_next_batch == is_next_batch
    assert session.va_batch_index == va_batch_index
    assert session.va_batch_count == va_batch_count
    assert session.is_next_va_batch == is_next_va_batch
    assert session.task_token == task_token
    assert session.server_region == server_region
    assert session.phase == phase
    assert session.direction == direction
    assert session.epoch_index == epoch_index
    assert session.is_next_epoch == is_next_epoch
    assert session.s3_bucket == s3_bucket
    assert session.shuffled_index_path == shuffled_index_path
    assert session.gradient_file_path == gradient_file_path


@pytest.fixture
def dataset(request):
    return Dataset(client=request.param)


@pytest.fixture
def vfl_sqs(request):
    name = request.param[0]
    region = request.param[1]
    client = boto3.client("sqs", region_name=region)
    client.create_queue(QueueName=name)
    yield VFLSQS(name=name, region=region)

    queue_url = client.get_queue_url(QueueName=name)["QueueUrl"]
    client.delete_queue(QueueUrl=queue_url)


@pytest.mark.parametrize(
    ("client", "vfl_sqs", "dataset", "model", "optimizer", "shuffled_index"),
    [
        (
            "1",
            ["vfl-test-us-west-1", "us-west-1"],
            "1",
            ClientModel(17, 4),
            torch.optim.Adam(ClientModel(17, 4).parameters(), lr=0.01),
            ShuffledIndex(),
        )
    ],
    indirect=["vfl_sqs", "dataset"],
)
def test_init_client_trainer(
    client, vfl_sqs, dataset, model, optimizer, shuffled_index
):
    trainer = ClientTrainer(
        client_id=client,
        queue=vfl_sqs,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        shuffled_index=shuffled_index,
    )
    assert trainer.client_id == client
    assert trainer.queue == vfl_sqs
    assert trainer.s3
    assert len(trainer.tr_uid) == len(dataset.tr_uid)
    assert len(trainer.tr_x) == len(dataset.tr_x)
    assert trainer.tr_xcols == dataset.tr_xcols
    assert len(trainer.va_uid) == len(dataset.va_uid)
    assert len(trainer.va_x) == len(dataset.va_x)
    assert trainer.va_xcols == dataset.va_xcols
    assert trainer.shuffled_index == shuffled_index
    assert trainer.model == model.to()
    assert trainer.optimizer == optimizer
    assert trainer.embed is None
    assert trainer.tmp_dir == f"tmp/{client}"


@pytest.fixture
def index(request):
    if request.param is None:
        yield {"Uri": None, "Object": None}
    with tempfile.TemporaryDirectory() as tmpdirname:
        sample_count = len(
            torch.FloatTensor(
                np.load(
                    "../server/functions/init_server/tr_uid.npy", allow_pickle=False
                )
            )
        )
        index = torch.randperm(sample_count)
        file_name = "VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.npy"
        local_path = f"{tmpdirname}/{file_name}"
        np.save(local_path, index.numpy(), allow_pickle=False)
        if request.param == "local":
            yield {"Uri": local_path, "Object": index}
        elif request.param == "s3":
            bucket_name = "".join(
                [
                    random.choice(string.ascii_lowercase + string.digits)
                    for i in range(20)
                ]
            )
            bucket = boto3.resource("s3").Bucket(bucket_name)
            bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
            bucket.upload_file(local_path, file_name)
            yield {"Uri": f"s3://{bucket.name}/{file_name}", "Object": index}

            bucket.objects.all().delete()
            bucket.delete()


@pytest.mark.parametrize(("index"), [None, "local", "s3"], indirect=True)
def test_shuffled_index(index):
    uri = index["Uri"]
    expected_object = index["Object"]

    shuffled_index = ShuffledIndex(uri=uri)
    if uri is None:
        assert shuffled_index.index is None
    else:
        assert shuffled_index.index.tolist() == expected_object.tolist()

    empty_shuffled_index = ShuffledIndex()
    assert empty_shuffled_index.uri is None
    assert empty_shuffled_index.index is None
    empty_shuffled_index.update_if_not_set(uri)
    if uri is None:
        assert empty_shuffled_index.uri is None
        assert empty_shuffled_index.index is None
    else:
        assert empty_shuffled_index.uri == uri
        assert empty_shuffled_index.index.tolist() == expected_object.tolist()


def test_client_model():
    model = ClientModel(17, 4).to()
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = f"{tmpdirname}/test-client-model.pt"
        model.save(file_path)
        saved_model = ClientModel(17, 4).to()
        saved_model.load_state_dict(torch.load(file_path))
        assert len(model.state_dict()) == len(saved_model.state_dict())


@pytest.fixture
def sqs():
    name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )
    region = "us-west-2"
    client = boto3.client("sqs", region_name=region)
    client.create_queue(QueueName=name)
    test_message = {
        "TaskName": "VFL-TAKS-YYYY-MM-DD-HH-mm-ss",
        "TaskToken": "1234567890",
        "Direction": "Forward",
        "Phase": "Training",
        "BatchSize": 1024,
        "BatchIndex": 5,
        "BatchCount": 6,
        "VaBatchIndex": 3,
        "VaBatchCount": 0,
        "IsNextBatch": True,
        "IsNextVaBatch": True,
        "EpochIndex": 3,
        "IsNextEpoch": True,
        "ShuffledIndexPath": "s3://test-vfl/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.npy",
    }
    message_body = json.dumps(test_message)
    queue_url = client.get_queue_url(QueueName=name)["QueueUrl"]
    client.send_message(QueueUrl=queue_url, MessageBody=message_body)

    yield {
        "Name": name,
        "Region": region,
        "Message": test_message,
    }

    client.delete_queue(QueueUrl=queue_url)


def test_vfl_sqs(sqs):
    name = sqs["Name"]
    region = sqs["Region"]
    expected_body = sqs["Message"]
    vfl_sqs = VFLSQS(name=name, region=region)
    assert vfl_sqs.name == name
    assert vfl_sqs.region == region

    message = vfl_sqs.receive_message()
    assert message.receipt_handle
    body = json.loads(message.body)
    assert body["TaskName"] == expected_body["TaskName"]
    assert body["TaskToken"] == expected_body["TaskToken"]
    assert body["Direction"] == expected_body["Direction"]
    assert body["Phase"] == expected_body["Phase"]
    assert body["BatchSize"] == expected_body["BatchSize"]
    assert body["BatchIndex"] == expected_body["BatchIndex"]
    assert body["BatchCount"] == expected_body["BatchCount"]
    assert body["VaBatchIndex"] == expected_body["VaBatchIndex"]
    assert body["VaBatchCount"] == expected_body["VaBatchCount"]
    assert body["IsNextBatch"] == expected_body["IsNextBatch"]
    assert body["IsNextVaBatch"] == expected_body["IsNextVaBatch"]
    assert body["EpochIndex"] == expected_body["EpochIndex"]
    assert body["IsNextEpoch"] == expected_body["IsNextEpoch"]
    assert body["ShuffledIndexPath"] == expected_body["ShuffledIndexPath"]
