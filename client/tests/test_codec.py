import pytest
import torch
import json
import random
from codec import SparseEncoder, SparseDecoder, SparseEncodedTensor


@pytest.mark.parametrize(
    ("encoded_embed"),
    [
        {
            "samples": random.randint(1, 100),
            "dims": random.randint(1, 100),
            "non_zero_values": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ),
            "nz_head": torch.randn(random.randint(1, 100), random.randint(1, 100)),
            "nz_tail": torch.randn(random.randint(1, 100), random.randint(1, 100)),
        },
        {
            "samples": random.randint(1, 100),
            "dims": random.randint(1, 100),
            "non_zero_values": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ),
            "nz_head": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ).long(),
            "nz_tail": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ).long(),
        },
        {
            "samples": random.randint(1, 100),
            "dims": random.randint(1, 100),
            "non_zero_values": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ),
            "nz_head": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ).tolist(),
            "nz_tail": torch.randn(
                random.randint(1, 100), random.randint(1, 100)
            ).tolist(),
        },
    ],
)
def test_sparse_encoded_tensor(encoded_embed):
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


def test_init_codec():
    assert SparseEncoder()
    assert SparseDecoder()


@pytest.mark.parametrize(
    ("tensor"),
    [
        torch.ones(1024, 1024),
        torch.zeros(1024, 1024),
        torch.randn(1024, 1024),
    ],
)
def test_encode(tensor):
    samples = tensor.shape[0]
    dims = tensor.shape[1]

    dst = tensor.detach().cpu().t().reshape(-1)
    length = len(dst)
    nz_pos = dst != 0
    non_zero_values = dst[nz_pos].to(torch.float16)

    nz_pos = nz_pos.char()
    ind = torch.arange(0, length)

    nz_cp = nz_pos - torch.cat((torch.CharTensor([0]), nz_pos[0:-1]), 0)
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
    sparse_embed = encoder.encode(tensor)

    assert expected["samples"] == sparse_embed.samples
    assert expected["dims"] == sparse_embed.dims
    assert torch.equal(expected["non_zero_values"], sparse_embed.non_zero_values)
    assert torch.equal(expected["nz_head"], sparse_embed.nz_head)
    assert torch.equal(expected["nz_tail"], sparse_embed.nz_tail)

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
