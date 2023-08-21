import pytest
import boto3
import json
import random
import string
import tempfile
import torch
import numpy as np
from functions.server_training.server_training import (
    S3Url,
    ShuffledIndex,
    ServerTrainer,
    ServerModel,
    TrainingSession,
    DataSet,
    Loss,
    Prediction,
    SparseDecoder,
    SparseEncoder,
    SparseEncodedTensor,
    SparseOptions,
)
from functions.init_server.serverinit import set_shuffled_index


def create_test_bucket():
    bucket_name = "".join(
        [random.choice(string.ascii_lowercase + string.digits) for i in range(20)]
    )

    bucket = boto3.resource("s3").Bucket(bucket_name)
    bucket.create(
        CreateBucketConfiguration={"LocationConstraint": "us-west-2"},
    )

    return bucket


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
        torch.FloatTensor(
            np.load(
                f"{dataset_dir}/tr_y.npy",
                allow_pickle=False,
            )
        )
    )
    assert (
        dataset.uid.dtype
        == torch.LongTensor(
            np.load(f"{dataset_dir}/tr_uid.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.uid) == len(
        torch.LongTensor(
            np.load(
                f"{dataset_dir}/tr_uid.npy",
                allow_pickle=False,
            )
        )
    )
    assert (
        dataset.va_label.dtype
        == torch.FloatTensor(
            np.load(f"{dataset_dir}/va_y.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.va_label) == len(
        torch.FloatTensor(
            np.load(
                f"{dataset_dir}/va_y.npy",
                allow_pickle=False,
            )
        )
    )
    assert (
        dataset.va_uid.dtype
        == torch.LongTensor(
            np.load(f"{dataset_dir}/va_uid.npy", allow_pickle=False)
        ).dtype
    )
    assert len(dataset.va_uid) == len(
        torch.LongTensor(
            np.load(
                f"{dataset_dir}/va_uid.npy",
                allow_pickle=False,
            )
        )
    )


@pytest.fixture
def shuffled_index_url() -> S3Url:
    bucket = create_test_bucket()

    shuffled_index_url = set_shuffled_index(
        "functions/init_server/tr_uid.npy",
        "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
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
def test_encoded_embed_params(request):
    samples = request.param["samples"]
    dims = request.param["dims"]
    length = samples * dims

    embed = torch.randn(samples, dims)
    dst = embed.detach().cpu().reshape(-1)
    nz_pos = (torch.FloatTensor(length).uniform_() > 0.8).bool()

    non_zero_values = dst[nz_pos].to(torch.float16)

    nz_pos_char = nz_pos.char()
    idx = torch.arange(0, length)

    nz_cp = nz_pos_char - torch.cat((torch.CharTensor([0]), nz_pos_char[0:-1]), 0)
    nz_head = idx[nz_cp == 1]
    nz_tail = idx[nz_cp == -1]

    return {
        "encoded_embed": {
            "samples": samples,
            "dims": dims,
            "non_zero_values": non_zero_values,
            "nz_head": nz_head,
            "nz_tail": nz_tail,
        },
        "nz_pos": nz_pos,
    }


@pytest.mark.parametrize(
    ("test_encoded_embed_params"),
    [
        {
            "samples": random.randint(1, 100),
            "dims": random.randint(1, 100),
        },
    ],
    indirect=True,
)
def test_sparse_encoded_tensor(test_encoded_embed_params):
    encoded_embed = test_encoded_embed_params["encoded_embed"]
    nz_pos = test_encoded_embed_params["nz_pos"]

    sparse_encoded_tensor = SparseEncodedTensor(encoded_embed)
    assert sparse_encoded_tensor.samples == encoded_embed["samples"]
    assert sparse_encoded_tensor.dims == encoded_embed["dims"]

    non_zero_values = encoded_embed["non_zero_values"]
    if type(non_zero_values) is list:
        non_zero_values = torch.Tensor(non_zero_values)

    assert torch.equal(sparse_encoded_tensor.non_zero_values, non_zero_values)

    nz_head = encoded_embed["nz_head"]
    if type(nz_head) is not torch.Tensor:
        nz_head = torch.Tensor(nz_head).long()
    else:
        nz_head = nz_head.long()

    assert torch.equal(sparse_encoded_tensor.nz_head, nz_head)

    nz_tail = encoded_embed["nz_tail"]
    if type(nz_tail) is not torch.Tensor:
        nz_tail = torch.Tensor(nz_tail).long()
    else:
        nz_tail = nz_tail.long()

    assert torch.equal(sparse_encoded_tensor.nz_tail, nz_tail)

    assert torch.equal(sparse_encoded_tensor.nz_pos, nz_pos)


def test_init_codec():
    assert SparseEncoder()
    assert SparseDecoder()


@pytest.mark.parametrize(
    ("tensor", "nz_pos"),
    [
        (
            torch.ones(1024, 1024),
            None,
        ),
        (
            torch.ones(1024, 1024),
            (torch.FloatTensor(1024 * 1024).uniform_() > 0.8).bool(),
        ),
        (
            torch.zeros(1024, 1024),
            None,
        ),
        (
            torch.zeros(1024, 1024),
            (torch.FloatTensor(1024 * 1024).uniform_() > 0.8).bool(),
        ),
        (
            torch.randn(1024, 1024),
            None,
        ),
        (
            torch.randn(1024, 1024),
            (torch.FloatTensor(1024 * 1024).uniform_() > 0.8).bool(),
        ),
    ],
)
def test_encode(tensor: torch.Tensor, nz_pos: torch.Tensor):
    samples = tensor.shape[0]
    dims = tensor.shape[1]

    dst = tensor.detach().cpu().t().reshape(-1)
    length = len(dst)
    if nz_pos is None:
        nz_pos = dst != 0
    non_zero_values = dst[nz_pos].to(torch.float16)

    nz_pos_char = nz_pos.char()
    ind = torch.arange(0, length)

    nz_cp = nz_pos_char - torch.cat((torch.CharTensor([0]), nz_pos_char[0:-1]), 0)
    nz_head = ind[nz_cp == 1]
    nz_tail = ind[nz_cp == -1]

    expected = {
        "samples": samples,
        "dims": dims,
        "non_zero_values": non_zero_values,
        "nz_head": nz_head,
        "nz_tail": nz_tail,
    }

    encoder = SparseEncoder()
    sparse_embed = encoder.encode(tensor, nz_pos)

    assert expected["samples"] == sparse_embed.samples
    assert expected["dims"] == sparse_embed.dims
    assert torch.equal(expected["non_zero_values"], sparse_embed.non_zero_values)
    assert torch.equal(expected["nz_head"], sparse_embed.nz_head)
    assert torch.equal(expected["nz_tail"], sparse_embed.nz_tail)
    assert torch.equal(nz_pos, sparse_embed.nz_pos)

    exported_sparse_embed = sparse_embed.export()

    assert expected["samples"] == exported_sparse_embed["samples"]
    assert expected["dims"] == exported_sparse_embed["dims"]
    assert (
        expected["non_zero_values"].tolist() == exported_sparse_embed["non_zero_values"]
    )
    assert expected["nz_head"].tolist() == exported_sparse_embed["nz_head"]
    assert expected["nz_tail"].tolist() == exported_sparse_embed["nz_tail"]

    assert json.dumps(exported_sparse_embed)


@pytest.mark.parametrize(
    ("tensor"),
    [
        torch.ones(1024, 1024),
        torch.zeros(1024, 1024),
        torch.randn(1024, 1024),
        torch.cat(
            (
                torch.randn(1024, 1024),
                torch.zeros(100, 1024),
                torch.randn(200, 1024),
            )
        ),
    ],
)
def test_decode(tensor: torch.FloatTensor):
    encoder = SparseEncoder()
    decoder = SparseDecoder()
    sparse_tensor = encoder.encode(tensor)
    decoded_tensor = decoder.decode(sparse_tensor)

    assert torch.equal(
        tensor.to(torch.float16),
        decoded_tensor.to(torch.float16),
    )

    exported_sparse_tensor = sparse_tensor.export()
    decoded_tensor = decoder.decode(SparseEncodedTensor(exported_sparse_tensor))

    assert torch.equal(
        tensor.to(torch.float16),
        decoded_tensor.to(torch.float16),
    )

    json_exported_sparse_tensor = sparse_tensor.export_as_json()
    decoded_tensor = decoder.decode(
        SparseEncodedTensor(json.loads(json_exported_sparse_tensor))
    )

    assert torch.equal(
        tensor.to(torch.float16),
        decoded_tensor.to(torch.float16),
    )


@pytest.fixture
def model_bucket():
    bucket = create_test_bucket()
    yield bucket

    bucket.objects.all().delete()
    bucket.delete()


def test_server_model(model_bucket):
    model = ServerModel(16, 1)
    s3_object = boto3.resource("s3").Object(
        model_bucket.name, "server/VFL-TASK-YYYY-MM-DD-HH-mm-ss-server-model.pt"
    )
    s3_url = model.save(s3_object)
    with tempfile.TemporaryDirectory() as tmpdirname:
        file_path = f"{tmpdirname}/{s3_url.file_name}"
        s3_object.download_file(file_path)
        saved_model = ServerModel(16, 1)
        saved_model.load_state_dict(torch.load(file_path))
        assert len(model.state_dict()) == len(saved_model.state_dict())


@pytest.fixture
def loss_object(request):
    bucket = create_test_bucket()
    task_name = request.param["TASK_NAME"]
    s3_key = request.param["S3_KEY"]
    loss = request.param["LOSS"]
    if loss is not None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/{task_name}-loss.json"
            with open(file_path, "w") as f:
                json.dump(loss, f)
            boto3.resource("s3").Object(bucket.name, s3_key).upload_file(file_path)

    yield boto3.resource("s3").Object(bucket.name, s3_key)

    bucket.objects.all().delete()
    bucket.delete()


@pytest.mark.parametrize(
    ("loss_object", "expected"),
    [
        (
            {
                "TASK_NAME": "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
                "LOSS": None,
                "S3_KEY": "common/VFL-TASK-YYYY-MM-DD-HH-mm-ss-loss.json",
            },
            {"total_tr_loss": 0, "total_va_loss": 0},
        ),
        (
            {
                "TASK_NAME": "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
                "LOSS": {
                    "total_tr_loss": 3.3729172945022583,
                    "total_va_loss": 1.1351569890975952,
                },
                "S3_KEY": "server/VFL-TASK-YYYY-MM-DD-HH-mm-ss-loss.json",
            },
            {"total_tr_loss": 3.3729172945022583, "total_va_loss": 1.1351569890975952},
        ),
    ],
    indirect=["loss_object"],
)
def test_loss(loss_object, expected):
    loss = Loss(loss_object)
    assert loss.total_tr_loss == 0
    assert loss.total_va_loss == 0

    loss.load()
    assert loss.total_tr_loss == expected["total_tr_loss"]
    assert loss.total_va_loss == expected["total_va_loss"]

    loss.total_tr_loss += 1
    loss.total_va_loss += 0.5

    new_loss_object = loss_object

    loss.save()
    new_loss = Loss(new_loss_object)
    new_loss.load()
    assert new_loss.total_tr_loss == loss.total_tr_loss
    assert new_loss.total_va_loss == loss.total_va_loss


@pytest.fixture
def prediction_test_params(request):
    shape = request.param
    bucket = create_test_bucket()
    s3_object = boto3.resource("s3").Object(
        bucket.name, "server/VFL-TASK-YYYY-MM-DD-HH-mm-ss-tr-pred.npy"
    )
    value = np.random.rand(shape[0], shape[1])
    with tempfile.TemporaryDirectory() as tmpdirname:
        np.save(f"{tmpdirname}/prediction.npy", value, allow_pickle=False)
        s3_object.upload_file(f"{tmpdirname}/prediction.npy")

    yield {
        "shape": shape,
        "s3_object": s3_object,
        "expected": value,
    }

    bucket.objects.all().delete()
    bucket.delete()


@pytest.mark.parametrize(
    "prediction_test_params",
    [
        (100, 200),
        (20000, 1),
        (1, 20000),
        (200, 100),
    ],
    indirect=True,
)
def test_prediction(prediction_test_params):
    shape = prediction_test_params["shape"]
    s3_obejct = prediction_test_params["s3_object"]
    expected = prediction_test_params["expected"]
    prediction = Prediction(shape, s3_obejct)
    assert np.all(prediction.value == np.zeros(shape=shape))

    prediction.load()
    assert np.all(prediction.value == expected)

    prediction.value = np.random.rand(shape[0], shape[1])
    prediction.save()
    new_prediction = Prediction(prediction.shape, prediction.s3_object)
    new_prediction.load()
    assert np.all(prediction.value == new_prediction.value)


@pytest.fixture
def bucket_name() -> str:
    bucket = create_test_bucket()

    set_shuffled_index(
        "functions/init_server/tr_uid.npy",
        "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
        bucket.name,
        prefix="common/",
    )

    yield bucket.name

    bucket.objects.all().delete()
    bucket.delete()


@pytest.fixture
def shuffled_index() -> ShuffledIndex:
    bucket = create_test_bucket()

    s3_url = set_shuffled_index(
        "functions/init_server/tr_uid.npy",
        "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
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
        "tr_pred",
        "va_pred",
        "loss",
        "sparse_options",
    ),
    [
        (
            "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
            4,
            0,
            0,
            1024,
            0,
            Prediction([29304, 1], s3_object=None),
            Prediction([3257, 1], s3_object=None),
            Loss(),
            SparseOptions(enabled=True, sparse_lambda=0.1),
        ),
        (
            "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
            4,
            0,
            0,
            1024,
            0,
            Prediction([29304, 1], s3_object=None),
            Prediction([3257, 1], s3_object=None),
            Loss(),
            SparseOptions(),
        ),
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
    tr_pred,
    va_pred,
    loss,
    sparse_options,
):
    session = TrainingSession(
        task_name=task_name,
        num_of_clients=num_of_clients,
        epoch_index=epoch_index,
        batch_index=batch_index,
        batch_size=batch_size,
        va_batch_index=va_batch_index,
        shuffled_index=shuffled_index,
        tr_pred=tr_pred,
        va_pred=va_pred,
        loss=loss,
        sparse_options=sparse_options,
    )
    assert session.task_name == task_name
    assert session.num_of_clients == num_of_clients
    assert session.epoch_index == epoch_index
    assert session.batch_index == batch_index
    assert session.batch_size == batch_size
    assert session.va_batch_index == va_batch_index
    assert session.shuffled_index.tolist() == shuffled_index.index.tolist()
    assert np.all(session.tr_pred.value == tr_pred.value)
    assert np.all(session.va_pred.value == va_pred.value)
    assert session.loss.total_tr_loss == loss.total_tr_loss
    assert session.loss.total_va_loss == loss.total_va_loss
    assert session.sparse_options == sparse_options


@pytest.mark.parametrize(
    (
        "task_name",
        "num_of_clients",
        "epoch_index",
        "batch_index",
        "batch_size",
        "va_batch_index",
        "tr_pred",
        "va_pred",
        "loss",
        "sparse_options",
    ),
    [
        (
            "VFL-TASK-YYYY-MM-DD-HH-mm-ss",
            4,
            0,
            0,
            1024,
            0,
            np.zeros([29304, 1]),
            np.zeros([3257, 1]),
            Loss(),
            SparseOptions(),
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
    tr_pred,
    va_pred,
    loss,
    shuffled_index,
    sparse_options,
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
        tr_pred=tr_pred,
        va_pred=va_pred,
        loss=loss,
        sparse_options=sparse_options,
    )
    model = ServerModel(4 * num_of_clients, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    server_trainer = ServerTrainer(
        training_session=training_session,
        s3_bucket=bucket_name,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
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
    assert server_trainer.gradients == dict()
    assert np.all(server_trainer.tr_pred == tr_pred)
    assert np.all(server_trainer.va_pred == va_pred)
    assert server_trainer.loss.total_tr_loss == loss.total_tr_loss
    assert server_trainer.loss.total_va_loss == loss.total_va_loss
    assert len(server_trainer.model.state_dict()) == len(model.state_dict())
    assert len(server_trainer.optimizer.state_dict()) == len(optimizer.state_dict())
    assert server_trainer.sparse_options == sparse_options
