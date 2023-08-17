import abc
import torch
import json


class SparseEncodedTensor:
    samples: int
    dims: int
    non_zero_values: torch.Tensor
    nz_head: torch.Tensor
    nz_tail: torch.Tensor

    def __init__(self, encoded_embed: dict) -> None:
        if "samples" in encoded_embed:
            self.samples = encoded_embed["samples"]
        if "dims" in encoded_embed:
            self.dims = encoded_embed["dims"]
        if "non_zero_values" in encoded_embed:
            self.non_zero_values = torch.Tensor(encoded_embed["non_zero_values"])
        if "nz_head" in encoded_embed:
            self.nz_head = torch.Tensor(encoded_embed["nz_head"]).long()
        if "nz_tail" in encoded_embed:
            self.nz_tail = torch.Tensor(encoded_embed["nz_tail"]).long()

    def export(self) -> dict:
        return {
            "samples": self.samples,
            "dims": self.dims,
            "non_zero_values": self.non_zero_values.tolist(),
            "nz_head": self.nz_head.tolist(),
            "nz_tail": self.nz_tail.tolist(),
        }

    def export_as_json(self) -> str:
        return json.dumps(
            {
                "samples": self.samples,
                "dims": self.dims,
                "non_zero_values": self.non_zero_values.tolist(),
                "nz_head": self.nz_head.tolist(),
                "nz_tail": self.nz_tail.tolist(),
            }
        )


class IEncoder(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def encode(self, tensor: torch.Tensor) -> SparseEncodedTensor:
        raise NotImplementedError()


class IDecoder(metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def decode(self, tensor: SparseEncodedTensor) -> torch.Tensor:
        raise NotImplementedError()


class SparseEncoder(IEncoder):
    def __init__(self) -> None:
        super().__init__()

    def encode(self, tensor: torch.Tensor) -> SparseEncodedTensor:
        samples = tensor.shape[0]
        dims = tensor.shape[1]

        dst = tensor.detach().t().reshape(-1)
        nz_pos = dst != 0
        length = len(dst)

        non_zero_values = dst[nz_pos].to(torch.float16)
        nz_pos = nz_pos.char()
        idx = torch.arange(0, length)

        nz_cp = nz_pos - torch.cat((torch.CharTensor([0]), nz_pos[0:-1]), 0)
        nz_head = idx[nz_cp == 1]
        nz_tail = idx[nz_cp == -1]

        return SparseEncodedTensor(
            {
                "samples": samples,
                "dims": dims,
                "non_zero_values": non_zero_values,
                "nz_head": nz_head,
                "nz_tail": nz_tail,
            }
        )


class SparseDecoder(IDecoder):
    def __init__(self) -> None:
        super().__init__()

    def decode(self, encoded_tensor: SparseEncodedTensor) -> torch.FloatTensor:
        samples = encoded_tensor.samples
        dims = encoded_tensor.dims
        length = samples * dims
        nz_head = encoded_tensor.nz_head
        nz_tail = encoded_tensor.nz_tail
        non_zero_values = encoded_tensor.non_zero_values

        nz_cp = torch.zeros(length, dtype=torch.int8)

        nz_cp[nz_head] = 1
        nz_cp[nz_tail] = -1

        nz_pos = torch.cumsum(nz_cp, dim=0).bool()

        tensor = torch.zeros(length)
        tensor[nz_pos] = non_zero_values.float()
        tensor = tensor.reshape((dims, samples)).t().to()

        return tensor
