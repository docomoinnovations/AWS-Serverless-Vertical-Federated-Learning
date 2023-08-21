import pytest
import torch
import numpy as np
import boto3
import tempfile
import random
import string
import json
from zipfile import ZipFile
from codec import SparseEncoder, SparseDecoder
from local_training import (
    S3Url,
    ClientTrainer,
    Dataset,
    Gradient,
    VFLSQS,
    ClientModel,
    ShuffledIndex,
)


def create_test_bucket():
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )

    return bucket


@pytest.fixture
def bucket():
    bucket = create_test_bucket()
    yield bucket

    bucket.objects.all().delete()
    bucket.delete()


@pytest.mark.parametrize(
    ("s3_url", "bucket", "key", "prefix", "file_name"),
    [
        ("s3://test/example.jpg", "test", "example.jpg", "", "example.jpg"),
        (
            "s3://test/prefix/shuffled-index.npy",
            "test",
            "prefix/shuffled-index.npy",
            "prefix/",
            "shuffled-index.npy",
        ),
        (
            "s3://test.example.com/prefix/shuffled-index.npy",
            "test.example.com",
            "prefix/shuffled-index.npy",
            "prefix/",
            "shuffled-index.npy",
        ),
        (
            "s3://test.example.com/prefix/test/shuffled-index.npy",
            "test.example.com",
            "prefix/test/shuffled-index.npy",
            "prefix/test/",
            "shuffled-index.npy",
        ),
    ],
)
def test_s3_url(s3_url, bucket, key, prefix, file_name):
    s3_url_obj = S3Url(s3_url)
    assert s3_url_obj.url == s3_url
    assert s3_url_obj.bucket == bucket
    assert s3_url_obj.key == key
    assert s3_url_obj.prefix == prefix
    assert s3_url_obj.file_name == file_name


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


@pytest.fixture
def gradient_test_params(request):
    shape = request.param["shape"]
    key = request.param["key"]
    encode = request.param["encode"]

    gradient_num = np.random.rand(shape[0], shape[1])
    expected = torch.FloatTensor(gradient_num)
    gradient = json.dumps(expected.tolist())

    if encode:
        encoder = SparseEncoder()
        gradient = encoder.encode(expected).export_as_json()

    bucket = create_test_bucket()
    s3_object = boto3.resource("s3").Object(
        bucket.name,
        key,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = f"{tmpdirname}/gradient.json"
        zip_file = f"{tmpdirname}/gradient.zip"
        with open(file=file_path, mode="w") as f:
            f.write(gradient)

        with ZipFile(zip_file, "w") as zipf:
            zipf.write(file_path, arcname=file_path.split("/")[-1])

        s3_object.upload_file(zip_file)

    yield {
        "s3_object": s3_object,
        "expected": expected,
        "encode": encode,
    }

    bucket.objects.all().delete()
    bucket.delete()


@pytest.mark.parametrize(
    "gradient_test_params",
    [
        {
            "shape": (9304, 4),
            "key": "client1/VFL-TASK-YYYY-MM-DD-HH-mm-ss-gradient-1.zip",
            "encode": False,
        },
        {
            "shape": (3257, 4),
            "key": "client2/VFL-TASK-YYYY-MM-DD-HH-mm-ss-gradient-2.zip",
            "encode": True,
        },
    ],
    indirect=True,
)
def test_gradient(gradient_test_params):
    s3_object = gradient_test_params["s3_object"]
    expected = gradient_test_params["expected"]
    encode = gradient_test_params["encode"]

    decoder = None
    if encode:
        decoder = SparseDecoder()
    gradient = Gradient(s3_object=s3_object, decoder=decoder)
    assert torch.equal(expected.to(torch.float16), gradient.value.to(torch.float16))


@pytest.fixture
def dataset(request):
    return Dataset(client=request.param)


@pytest.mark.parametrize(
    (
        "client",
        "dataset",
        "model",
        "optimizer",
        "shuffled_index",
    ),
    [
        (
            "1",
            "1",
            ClientModel(17, 4),
            torch.optim.Adam(ClientModel(17, 4).parameters(), lr=0.01),
            ShuffledIndex(),
        )
    ],
    indirect=[
        "dataset",
    ],
)
def test_client_trainer(
    client,
    dataset,
    model,
    optimizer,
    shuffled_index,
    bucket,
):
    s3_best_model_object = boto3.resource("s3").Object(
        bucket.name,
        f"model/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-client-model-{client}-best.pt",
    )

    trainer = ClientTrainer(
        client_id=client,
        dataset=dataset,
        model=model,
        optimizer=optimizer,
        shuffled_index=shuffled_index,
    )
    assert trainer.client_id == client
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
    assert trainer.va_embed is None

    trainer.commit_model()
    trainer.save_model(s3_best_model_object)
    with tempfile.TemporaryDirectory() as tmpdirname:
        path = f"{tmpdirname}/client-model.pt"
        s3_best_model_object.download_file(path)
        saved_model = model
        saved_model.load_state_dict(torch.load(path))

        assert saved_model == model


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
            key = f"common/{file_name}"
            bucket = boto3.resource("s3").Bucket(bucket_name)
            bucket.create(CreateBucketConfiguration={"LocationConstraint": "us-west-2"})
            bucket.upload_file(local_path, key)
            yield {"Uri": f"s3://{bucket.name}/{key}", "Object": index}

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
    url = client.get_queue_url(QueueName=name)["QueueUrl"]
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
        "EpochCount": 10,
        "IsNextEpoch": True,
        "ShuffledIndexPath": "s3://test-vfl/VFL-TAKS-YYYY-MM-DD-HH-mm-ss-shuffled-index.npy",
    }
    message_body = json.dumps(test_message)
    queue_url = client.get_queue_url(QueueName=name)["QueueUrl"]
    client.send_message(QueueUrl=queue_url, MessageBody=message_body)

    yield {
        "Name": name,
        "Region": region,
        "Url": url,
        "Message": test_message,
    }

    client.delete_queue(QueueUrl=queue_url)


def test_vfl_sqs(sqs):
    name = sqs["Name"]
    region = sqs["Region"]
    url = sqs["Url"]
    expected_body = sqs["Message"]
    vfl_sqs = VFLSQS(name=name, region=region)
    assert vfl_sqs.name == name
    assert vfl_sqs.region == region
    assert vfl_sqs.url == url

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
    assert body["EpochCount"] == expected_body["EpochCount"]
    assert body["IsNextEpoch"] == expected_body["IsNextEpoch"]
    assert body["ShuffledIndexPath"] == expected_body["ShuffledIndexPath"]
