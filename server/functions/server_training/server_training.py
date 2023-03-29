import numpy as np
import torch
import random
import tempfile
import boto3
import json
from typing import Dict
from urllib.parse import urlparse
import torch.nn.functional as F
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
    def __init__(self, url: S3Url) -> None:
        self.url = url

        s3_object = boto3.resource("s3").Object(url.bucket, url.key)
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/embed.npy"
            s3_object.download_file(file_path)
            self.value = torch.FloatTensor(
                np.load(
                    file=file_path,
                    allow_pickle=False,
                )
            )


class Gradient:
    def __init__(self, value: torch.FloatTensor, s3_object) -> None:
        self.value = value
        self.s3_object = s3_object

    def save(self) -> S3Url:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/gradient.npy"
            np.save(file=file_path, arr=self.value.numpy(), allow_pickle=False)
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
    def __init__(self, hidden_size, out_size):
        super(ServerModel, self).__init__()
        self.h2h = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.h2o = torch.nn.Linear(hidden_size // 2, out_size)

        set_seed(seed)
        torch.nn.init.xavier_uniform_(self.h2h.weight.data)
        torch.nn.init.ones_(self.h2h.bias.data)
        set_seed(seed)
        torch.nn.init.xavier_uniform_(self.h2o.weight.data)
        torch.nn.init.ones_(self.h2o.bias.data)

    def forward(self, h):
        h = self.h2h(h)
        h = F.relu(h)
        o = self.h2o(h)
        return o

    def save(self, s3_object) -> S3Url:
        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f"{tmpdirname}/{s3_object.bucket_name}"
            torch.save(self.state_dict(), file_path)
            s3_object.upload_file(file_path)
        return S3Url(f"s3://{s3_object.bucket_name}/{s3_object.key}")


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
        for client_id in self.client_ids:
            embed_tuple = (*embed_tuple, self.embeds[client_id].value)
        embed = torch.cat(embed_tuple, 1)
        embed.requires_grad_(True)

        pred_y = self.model(embed)
        loss = self.criterion(pred_y, batch_y)
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
    # Extract message
    print(event)
    s3_bucket = event["VFLBucket"]
    phase = event["Phase"]
    batch_size = int(event["BatchSize"])
    input_items = event["InputItems"]
    num_of_clients = len(input_items)
    batch_index = int(input_items[0]["BatchIndex"])
    batch_count = int(input_items[0]["BatchCount"])
    va_batch_index = int(input_items[0]["VaBatchIndex"])
    va_batch_count = int(input_items[0]["VaBatchCount"])
    is_next_batch = bool(input_items[0]["IsNextBatch"])
    is_next_va_batch = bool(input_items[0]["IsNextVaBatch"])
    epoch_index = int(input_items[0]["EpochIndex"])
    is_next_epoch = bool(input_items[0]["IsNextEpoch"])
    task_name = input_items[0]["TaskName"]
    shuffled_index_path = input_items[0]["ShuffledIndexPath"]

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
    )

    model = ServerModel(4 * num_of_clients, 1)
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
                    "VaBatchIndex": va_batch_index,
                    "VaBatchCount": va_batch_count,
                    "IsNextBatch": is_next_batch,
                    "IsNextVaBatch": is_next_va_batch,
                    "EpochIndex": epoch_index,
                    "IsNextEpoch": is_next_epoch,
                    "ShuffledIndexPath": shuffled_index_path,
                    "SqsUrl": queue_url,
                }
            )

        print(json.dumps(response))
        return response

    # Set embed and gradient
    for input_item in input_items:
        client_id = input_item["MemberId"]
        url = S3Url(input_item["EmbedFile"])
        embed = Embed(url=url)
        server_trainer.set_embed(
            client_id=client_id,
            embed=embed,
        )

        gradient_object = boto3.resource("s3").Object(
            s3_bucket, f"{url.prefix}{task_name}-gradient-{client_id}.npy"
        )
        gradient_value = torch.FloatTensor(np.zeros(embed.value.shape))
        gradient = Gradient(value=gradient_value, s3_object=gradient_object)
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
                    "VaBatchIndex": va_batch_index,
                    "VaBatchCount": va_batch_count,
                    "IsNextBatch": is_next_batch,
                    "IsNextVaBatch": is_next_va_batch,
                    "EpochIndex": epoch_index,
                    "IsNextEpoch": is_next_epoch,
                    "GradientFile": gradient_files[member_id],
                    "ShuffledIndexPath": shuffled_index_path,
                    "SqsUrl": queue_url,
                    "TrLoss": server_trainer.get_tr_loss(),
                    "TrAuc": server_trainer.get_tr_auc(),
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
                    "VaBatchIndex": va_batch_index,
                    "VaBatchCount": va_batch_count,
                    "IsNextBatch": is_next_batch,
                    "IsNextVaBatch": is_next_va_batch,
                    "EpochIndex": epoch_index,
                    "IsNextEpoch": is_next_epoch,
                    "ShuffledIndexPath": shuffled_index_path,
                    "SqsUrl": queue_url,
                    "VaLoss": server_trainer.get_va_loss(),
                    "VaAuc": server_trainer.get_va_auc(),
                }
            )

    print(json.dumps(response))
    return response
