import abc
import torch
import json
import numpy as np
import random
import tempfile
import boto3
from typing import Dict, Optional
from urllib.parse import urlparse
from sklearn.metrics import roc_auc_score


class S3Url:
    def __init__(self, url) -> None:
        self.url = url
        parsed_url = urlparse(url, allow_fragments=False)
        self.bucket = parsed_url.netloc
        self.key = parsed_url.path.lstrip("/")
        self.file_name = self.key.split("/")[-1]
        self.prefix = self.key[: -len(self.file_name)]


class DataSet:
    def __init__(
        self,
        label="tr_y.npy",
        uid="tr_uid.npy",
        va_label="va_y.npy",
        va_uid="va_uid.npy",
    ) -> None:
        self.label = torch.FloatTensor(np.load(label, allow_pickle=False))
        self.uid = torch.LongTensor(np.load(uid, allow_pickle=False))
        self.va_label = torch.FloatTensor(np.load(va_label, allow_pickle=False))
        self.va_uid = torch.LongTensor(np.load(va_uid, allow_pickle=False))


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


class ShuffledIndex:
    def __init__(self, s3_url: S3Url):
        self.s3_url = s3_url
        s3_object = boto3.resource("s3").Object(s3_url.bucket, s3_url.key)
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/{s3_url.file_name}"
            s3_object.download_file(file_path)
            self.index = torch.LongTensor(np.load(file_path, allow_pickle=False))


class Prediction:
    def __init__(self, shape, s3_object) -> None:
        self.shape = shape
        self.s3_object = s3_object
        self.value = np.zeros(shape=shape)

    def load(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/prediction.npy"
            self.s3_object.download_file(file_path)
            value = np.load(file=file_path, allow_pickle=False)
            if value.shape != self.shape:
                print(
                    f"The shape of prediction is unexpected: expected({self.shape}), actual({value.shape})"
                )
            self.value = np.load(file=file_path, allow_pickle=False)

    def save(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/prediction.npy"
            np.save(file_path, self.value, allow_pickle=False)
            self.s3_object.upload_file(file_path)


class Embed:
    def __init__(self, url: S3Url, decoder: Optional[IDecoder] = None) -> None:
        self.url = url
        s3_object = boto3.resource("s3").Object(url.bucket, url.key)

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/embed.json"
            s3_object.download_file(file_path)
            embed = None
            with open(file=file_path, mode="r") as f:
                embed = json.load(f)
            if decoder:
                encoded_embed = SparseEncodedTensor(embed)
                self.value = decoder.decode(encoded_embed)
            else:
                self.value = torch.Tensor(embed)


class Gradient:
    def __init__(
        self,
        value: torch.Tensor,
        s3_object,
        encoder: Optional[IEncoder] = None,
    ) -> None:
        self.value = value
        self.s3_object = s3_object
        self.encoder = encoder

    def save(self) -> S3Url:
        gradient = self.value.tolist()
        if self.encoder:
            encoded_gradient = self.encoder.encode(self.value)
            gradient = encoded_gradient.export()
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/gradient.json"
            with open(file=file_path, mode="w") as f:
                json.dump(gradient, f)
            self.s3_object.upload_file(file_path)
        return S3Url(f"s3://{self.s3_object.bucket_name}/{self.s3_object.key}")


class Loss:
    def __init__(self, s3_object=None) -> None:
        self.total_tr_loss = 0
        self.total_va_loss = 0
        self.s3_object = s3_object

    def load(self):
        if self.s3_object is None:
            return

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/loss.json"
            try:
                self.s3_object.download_file(file_path)
            except Exception as e:
                print(e)
                return
            with open(file_path, "r") as f:
                loss = json.load(f)
                self.total_tr_loss = loss["total_tr_loss"]
                self.total_va_loss = loss["total_va_loss"]

    def save(self):
        if self.s3_object is None:
            return

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/loss.json"
            with open(file_path, "w") as f:
                json.dump(
                    {
                        "total_tr_loss": self.total_tr_loss,
                        "total_va_loss": self.total_va_loss,
                    },
                    f,
                )
            try:
                self.s3_object.upload_file(file_path)
            except Exception as e:
                print(e)


seed = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ServerModel(torch.nn.Module):
    def __init__(self, hidden_size, out_size, hidden_layer_count=1):
        super(ServerModel, self).__init__()
        self.hidden_layer_count = hidden_layer_count
        self.h2h, hidden_size = self._hidden_layers(hidden_size)
        self.h2o = torch.nn.Linear(hidden_size, out_size)

        set_seed(seed)
        torch.nn.init.xavier_uniform_(self.h2o.weight.data)
        torch.nn.init.ones_(self.h2o.bias.data)

    def _hidden_layers(self, hidden_size):
        layers = []
        for i in range(self.hidden_layer_count):
            h2h = torch.nn.Linear(hidden_size, hidden_size // 2)
            layers.append(h2h)

            set_seed(seed)
            torch.nn.init.xavier_uniform_(h2h.weight.data)
            torch.nn.init.ones_(h2h.bias.data)

            layers.append(torch.nn.ReLU())
            hidden_size = hidden_size // 2

        return torch.nn.Sequential(*layers), hidden_size

    def forward(self, h):
        h = self.h2h(h)
        output = self.h2o(h)
        return output

    def save(self, s3_object) -> S3Url:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/{s3_object.bucket_name}"
            torch.save(self.state_dict(), file_path)
            s3_object.upload_file(file_path)
        return S3Url(f"s3://{s3_object.bucket_name}/{s3_object.key}")


class SparseOptions:
    def __init__(self, enabled: bool = False, sparse_lambda: float = 0.1) -> None:
        self.enabled = enabled
        self.sparse_lambda = sparse_lambda

    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__


class TrainingSession:
    def __init__(
        self,
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
        sparse_options=SparseOptions(),
    ) -> None:
        self.task_name = task_name
        self.num_of_clients = num_of_clients
        self.epoch_index = epoch_index
        self.batch_index = batch_index
        self.batch_size = batch_size
        self.va_batch_index = va_batch_index
        self.shuffled_index = shuffled_index.index
        self.tr_pred = tr_pred
        self.va_pred = va_pred
        self.loss = loss
        self.sparse_options = sparse_options


class ServerTrainer:
    def __init__(
        self,
        training_session: TrainingSession,
        s3_bucket: str,
        model: ServerModel,
        optimizer: torch.optim.Adam,
        dataset: DataSet,
        embeds: Dict[str, Embed] = dict(),
        gradients: Dict[str, Gradient] = dict(),
    ) -> None:
        self.task_name = training_session.task_name
        self.num_of_clients = training_session.num_of_clients
        self.client_ids = [str(i + 1) for i in range(self.num_of_clients)]
        self.epoch_index = training_session.epoch_index
        self.batch_index = training_session.batch_index
        self.batch_size = training_session.batch_size
        self.va_batch_index = training_session.va_batch_index
        self.shuffled_index = training_session.shuffled_index
        self.loss = training_session.loss
        self.sparse_options = training_session.sparse_options
        self.s3_bucket = s3_bucket
        self.embeds = embeds
        self.gradients = gradients

        self.tr_uid = dataset.uid
        self.tr_y = dataset.label
        self.va_uid = dataset.va_uid
        self.va_y = dataset.va_label
        self.tr_true = self.tr_y[self.shuffled_index, :]
        self.va_true = self.va_y
        self.tr_pred = training_session.tr_pred
        self.va_pred = training_session.va_pred

        self.pos_weight = (self.tr_y.shape[0] - self.tr_y.sum()) / self.tr_y.sum()
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        self.model = model
        self.optimizer = optimizer

    def set_embed(self, client_id: str, embed: Embed) -> None:
        self.embeds[client_id] = embed

    def set_gradient(self, client_id: str, gradient: Gradient) -> None:
        self.gradients[client_id] = gradient

    def save_gradient(self, client_id: str) -> S3Url:
        return self.gradients[client_id].save()

    def save_model(self, s3_object) -> S3Url:
        return self.model.save(s3_object)

    def save_optimizer(self, s3_object) -> S3Url:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/optimizer.pt"
            torch.save(self.optimizer.state_dict(), file_path)
            s3_object.upload_file(file_path)
        return S3Url(f"s3://{s3_object.bucket_name}/{s3_object.key}")

    def save_loss(self) -> None:
        self.loss.save()

    def save_tr_pred(self) -> None:
        self.tr_pred.save()

    def save_va_pred(self) -> None:
        self.va_pred.save()

    def train(self) -> None:
        tr_sample_count = len(self.tr_uid)
        head = self.batch_index * self.batch_size
        tail = min(head + self.batch_size, tr_sample_count)
        si = self.shuffled_index[head:tail]

        self.model.train()
        batch_y = self.tr_y[si, :]
        self.optimizer.zero_grad()
        embed_tuple = ()

        sparse_embed_loss = 0
        for client_id in self.client_ids:
            embed_tuple = (*embed_tuple, self.embeds[client_id].value)
            if self.sparse_options.enabled:
                sparse_embed_loss = (
                    sparse_embed_loss
                    + torch.norm(self.embeds[client_id].value, p=1, dim=1).mean()
                )
        embed = torch.cat(embed_tuple, 1)
        embed.requires_grad_(True)

        pred_y = self.model(embed)
        loss = self.criterion(pred_y, batch_y)

        ## L1-norm
        if self.sparse_options.enabled:
            loss = loss + self.sparse_options.sparse_lambda * (
                sparse_embed_loss
            ) / float(self.num_of_clients)
        loss.backward()
        self.optimizer.step()
        self.loss.total_tr_loss += loss.item()
        for i, client_id in enumerate(self.client_ids):
            e_head = i * 4
            e_tail = (i + 1) * 4
            self.gradients[client_id].value = embed.grad[:, e_head:e_tail].cpu()

        self.tr_pred.value[head:tail, :] = torch.sigmoid(pred_y).detach().cpu().numpy()

    def validate(self) -> None:
        va_sample_count = len(self.va_uid)
        head = self.va_batch_index * self.batch_size
        tail = min(head + self.batch_size, va_sample_count)

        self.model.eval()
        batch_y = self.va_y[head:tail, :]
        embed_tuple = ()
        for client_id in self.client_ids:
            embed_tuple = (*embed_tuple, self.embeds[client_id].value)
        embed = torch.cat(embed_tuple, 1)

        pred_y = self.model(embed)
        loss = self.criterion(pred_y, batch_y)
        self.loss.total_va_loss += loss.item()
        self.va_pred.value[head:tail, :] = torch.sigmoid(pred_y).detach().cpu().numpy()

    def get_tr_loss(self) -> float:
        return self.loss.total_tr_loss / (self.batch_index + 1)

    def get_tr_auc(self) -> float:
        return roc_auc_score(self.tr_true, self.tr_pred.value)

    def get_va_loss(self) -> float:
        return self.loss.total_va_loss / (self.va_batch_index + 1)

    def get_va_auc(self) -> float:
        return roc_auc_score(self.va_true, self.va_pred.value)


def lambda_handler(event, context):
    print(event)

    phase = event["Phase"]
    input_items = event["InputItems"]
    num_of_clients = len(input_items)
    s3_bucket = input_items[0]["VFLBucket"]
    batch_size = input_items[0]["BatchSize"]
    batch_index = int(input_items[0]["BatchIndex"])
    batch_count = int(input_items[0]["BatchCount"])
    epoch_count = int(input_items[0]["EpochCount"])
    va_batch_index = int(input_items[0]["VaBatchIndex"])
    va_batch_count = int(input_items[0]["VaBatchCount"])
    is_next_batch = bool(input_items[0]["IsNextBatch"])
    is_next_va_batch = bool(input_items[0]["IsNextVaBatch"])
    epoch_index = int(input_items[0]["EpochIndex"])
    is_next_epoch = bool(input_items[0]["IsNextEpoch"])
    task_name = input_items[0]["TaskName"]
    shuffled_index_path = input_items[0]["ShuffledIndexPath"]
    patience = input_items[0]["Patience"]
    sparse_encoding = bool(input_items[0]["SparseEncoding"])
    sparse_lambda = float(input_items[0]["SparseLambda"])
    vfl_bucket = input_items[0]["VFLBucket"]

    # Initialize object stored on S3 bucket
    s3_model_object = boto3.resource("s3").Object(
        s3_bucket, f"server/{task_name}-server-model.pt"
    )
    s3_optimizer_object = boto3.resource("s3").Object(
        s3_bucket, f"server/{task_name}-optimizer.pt"
    )
    s3_tr_pred_object = boto3.resource("s3").Object(
        s3_bucket, f"server/{task_name}-tr-pred.npy"
    )
    s3_va_pred_object = boto3.resource("s3").Object(
        s3_bucket, f"server/{task_name}-va-pred.npy"
    )
    s3_loss_object = boto3.resource("s3").Object(
        s3_bucket, f"server/{task_name}-loss.json"
    )
    s3_best_model_object = boto3.resource("s3").Object(
        s3_bucket, f"model/{task_name}-server-model-best.pt"
    )

    dataset = DataSet()

    tr_pred = Prediction(
        shape=dataset.label.shape,
        s3_object=s3_tr_pred_object,
    )
    va_pred = Prediction(
        shape=dataset.va_label.shape,
        s3_object=s3_va_pred_object,
    )
    loss = Loss(s3_object=s3_loss_object)

    if batch_index > 0:
        loss.load()
        tr_pred.load()

    if va_batch_index > 0:
        va_pred.load()

    session = TrainingSession(
        task_name=task_name,
        num_of_clients=num_of_clients,
        epoch_index=epoch_index,
        batch_index=batch_index,
        batch_size=batch_size,
        va_batch_index=va_batch_index,
        shuffled_index=ShuffledIndex(S3Url(shuffled_index_path)),
        tr_pred=tr_pred,
        va_pred=va_pred,
        loss=loss,
        sparse_options=SparseOptions(
            enabled=sparse_encoding, sparse_lambda=sparse_lambda
        ),
    )

    model = ServerModel(4 * num_of_clients, 1)
    # Define encoder and decoder for embedding and gradient
    encoder = None
    decoder = None
    if sparse_encoding:
        encoder = SparseEncoder()
        decoder = SparseDecoder()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    if epoch_index != 0 or batch_index != 0:
        with tempfile.TemporaryDirectory() as tmpdirname:
            s3_model_file_name = s3_model_object.key.split("/")[-1]
            s3_model_object.download_file(f"{tmpdirname}/{s3_model_file_name}")
            model.load_state_dict(torch.load(f"{tmpdirname}/{s3_model_file_name}"))

            s3_optimizer_file_name = s3_optimizer_object.key.split("/")[-1]
            s3_optimizer_object.download_file(f"{tmpdirname}/{s3_optimizer_file_name}")
            optimizer.load_state_dict(
                torch.load(f"{tmpdirname}/{s3_optimizer_file_name}")
            )

    server_trainer = ServerTrainer(
        training_session=session,
        s3_bucket=s3_bucket,
        model=model,
        optimizer=optimizer,
        dataset=dataset,
    )

    # Save model and return
    if phase == "Save":
        # Save best model
        server_trainer.save_model(s3_best_model_object)

        # Generate response for the next Map step
        response = []
        for input_item in input_items:
            queue_url = input_item["SqsUrl"]
            response.append(
                {
                    "TaskName": task_name,
                    "BatchIndex": batch_index,
                    "BatchCount": batch_count,
                    "BatchSize": batch_size,
                    "EpochCount": epoch_count,
                    "EpochIndex": epoch_index,
                    "Patience": patience,
                    "SparseEncoding": sparse_encoding,
                    "SparseLambda": sparse_lambda,
                    "VaBatchIndex": va_batch_index,
                    "VaBatchCount": va_batch_count,
                    "IsNextBatch": is_next_batch,
                    "IsNextVaBatch": is_next_va_batch,
                    "IsNextEpoch": is_next_epoch,
                    "ShuffledIndexPath": shuffled_index_path,
                    "SqsUrl": queue_url,
                    "VFLBucket": vfl_bucket,
                }
            )

        print(json.dumps(response))
        return response

    # Set embed and gradient
    for input_item in input_items:
        client_id = input_item["MemberId"]
        url = S3Url(input_item["EmbedFile"])
        embed = Embed(url=url, decoder=decoder)
        server_trainer.set_embed(
            client_id=client_id,
            embed=embed,
        )

        gradient_object = boto3.resource("s3").Object(
            s3_bucket,
            f"{url.prefix}{task_name}-gradient-{client_id}.json",
        )
        gradient_value = torch.FloatTensor(np.zeros(embed.value.shape))
        gradient = Gradient(
            value=gradient_value,
            s3_object=gradient_object,
            encoder=encoder,
        )
        server_trainer.set_gradient(client_id=client_id, gradient=gradient)

    response = None

    # Training
    if phase == "Training":
        # Train model
        server_trainer.train()
        # Save parameters
        server_trainer.save_loss()
        server_trainer.save_tr_pred()

        gradient_files: Dict[str, str] = dict()

        for input_item in input_items:
            client_id = input_item["MemberId"]
            url = server_trainer.save_gradient(client_id=client_id)
            gradient_files[client_id] = url.url

        # Save model and optimizer
        server_trainer.save_model(s3_model_object)
        server_trainer.save_optimizer(s3_object=s3_optimizer_object)

        # Generate response for the next Map step
        response = []
        for input_item in input_items:
            queue_url = input_item["SqsUrl"]
            member_id = input_item["MemberId"]
            response.append(
                {
                    "TaskName": task_name,
                    "BatchIndex": batch_index,
                    "BatchCount": batch_count,
                    "BatchSize": batch_size,
                    "EpochCount": epoch_count,
                    "EpochIndex": epoch_index,
                    "SparseEncoding": sparse_encoding,
                    "SparseLambda": sparse_lambda,
                    "IsNextBatch": is_next_batch,
                    "IsNextVaBatch": is_next_va_batch,
                    "IsNextEpoch": is_next_epoch,
                    "GradientFile": gradient_files[member_id],
                    "Patience": patience,
                    "ShuffledIndexPath": shuffled_index_path,
                    "SqsUrl": queue_url,
                    "TrLoss": server_trainer.get_tr_loss(),
                    "TrAuc": server_trainer.get_tr_auc(),
                    "VaBatchIndex": va_batch_index,
                    "VaBatchCount": va_batch_count,
                    "VFLBucket": vfl_bucket,
                }
            )
    elif phase == "Validation":
        # Validate model
        server_trainer.validate()

        # Save parameters
        server_trainer.save_loss()
        server_trainer.save_va_pred()

        # Generate response for the next Map step
        response = []
        for input_item in input_items:
            queue_url = input_item["SqsUrl"]
            response.append(
                {
                    "TaskName": task_name,
                    "BatchIndex": batch_index,
                    "BatchCount": batch_count,
                    "BatchSize": batch_size,
                    "EpochCount": epoch_count,
                    "EpochIndex": epoch_index,
                    "VaBatchIndex": va_batch_index,
                    "VaBatchCount": va_batch_count,
                    "Patience": patience,
                    "SparseEncoding": sparse_encoding,
                    "SparseLambda": sparse_lambda,
                    "IsNextBatch": is_next_batch,
                    "IsNextVaBatch": is_next_va_batch,
                    "IsNextEpoch": is_next_epoch,
                    "ShuffledIndexPath": shuffled_index_path,
                    "SqsUrl": queue_url,
                    "VaLoss": server_trainer.get_va_loss(),
                    "VaAuc": server_trainer.get_va_auc(),
                    "VFLBucket": vfl_bucket,
                }
            )

    print(json.dumps(response))
    return response
