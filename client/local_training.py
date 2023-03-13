import json
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import random
import boto3

client_configs = {
    "1": "config/1.json",
    "2": "config/2.json",
    "3": "config/3.json",
    "4": "config/4.json",
}


class Dataset:
    def __init__(self, client) -> None:
        self.tr_uid = torch.load(f"dataset/client{client}/tr_uid.pt")
        self.tr_x = torch.load(f"dataset/client{client}/tr_x.pt")
        self.tr_xcols = torch.load(f"dataset/client{client}/cols.pt")
        self.va_uid = torch.load(f"dataset/client{client}/va_uid.pt")
        self.va_x = torch.load(f"dataset/client{client}/va_x.pt")
        self.va_xcols = self.tr_xcols
        self.tr_sample_count = len(self.tr_uid)
        self.va_sample_count = len(self.va_uid)


seed = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


class ClientModel(torch.nn.Module):
    def __init__(self, in_size, hidden_size):
        super(ClientModel, self).__init__()
        self.i2h = torch.nn.Linear(in_size, hidden_size)

        set_seed(seed)
        torch.nn.init.xavier_uniform_(self.i2h.weight.data)
        torch.nn.init.ones_(self.i2h.bias.data)

    def forward(self, x):
        h = F.relu(self.i2h(x))
        return h

    def save(self, file_path: str):
        torch.save(self.state_dict(), file_path)


class VFLSQS:
    def __init__(self, name, region) -> None:
        self.name = name
        self.region = region

        client = boto3.client("sqs", region_name=region)
        queue_url = client.get_queue_url(QueueName=self.name)["QueueUrl"]
        self.sqs = boto3.resource("sqs", region_name=region).Queue(queue_url)

    def receive_message(self):
        messages = self.sqs.receive_messages(
            AttributeNames=["SentTimestamp"],
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20,
        )
        if len(messages) > 0:
            return messages[0]
        else:
            return None


class TrainingSession:
    def __init__(
        self,
        task_name,
        batch_size=None,
        batch_index=None,
        batch_count=None,
        is_next_batch=None,
        va_batch_index=None,
        va_batch_count=None,
        is_next_va_batch=None,
        task_token=None,
        server_region=None,
        phase=None,
        direction=None,
        epoch_index=None,
        is_next_epoch=False,
        s3_bucket=None,
        shuffled_index_path=None,
        gradient_file_path=None,
    ) -> None:
        self.task_name = task_name
        self.batch_size = batch_size
        self.batch_index = batch_index
        self.batch_count = batch_count
        self.is_next_batch = is_next_batch
        self.va_batch_index = va_batch_index
        self.va_batch_count = va_batch_count
        self.is_next_va_batch = is_next_va_batch
        self.epoch_index = epoch_index
        self.is_next_epoch = is_next_epoch
        self.s3_bucket = s3_bucket
        self.shuffled_index_path = shuffled_index_path
        self.task_token = task_token
        self.server_region = server_region
        self.phase = phase
        self.direction = direction
        self.gradient_file_path = gradient_file_path


class ShuffledIndex:
    def __init__(self, uri: str = None) -> None:
        self.update(uri)

    def update(self, uri: str):
        self.uri = uri
        if uri is None:
            self.index = None
        elif uri.startswith("s3://"):
            bucket_name = uri.split("/")[2]
            key = "/".join(uri.split("/")[3:])
            bucket = boto3.resource("s3").Bucket(bucket_name)
            with tempfile.TemporaryDirectory() as tmpdirname:
                bucket.download_file(key, f"{tmpdirname}/{key}")
                self.index = torch.load(f"{tmpdirname}/{key}")
        else:
            self.index = torch.load(uri)

    def update_if_not_set(self, uri: str):
        if self.index is None:
            self.update(uri=uri)


class ClientTrainer:
    def __init__(
        self,
        client_id: str,
        queue: VFLSQS,
        dataset: Dataset,
        model: ClientModel,
        optimizer: torch.optim.Adam,
        shuffled_index: ShuffledIndex,
    ) -> None:
        self.client_id = client_id
        self.queue = queue
        self.s3 = boto3.resource("s3")
        self.tr_uid = dataset.tr_uid
        self.tr_x = dataset.tr_x
        self.tr_xcols = dataset.tr_xcols
        self.va_uid = dataset.va_uid
        self.va_x = dataset.va_x
        self.va_xcols = dataset.va_xcols

        self.shuffled_index = shuffled_index

        self.model = model.to()
        self.optimizer = optimizer
        self.embed = None

        self.tmp_dir = f"tmp/{self.client_id}"
        Path(f"tmp/{self.client_id}").mkdir(parents=True, exist_ok=True)

    def start(self):
        while True:
            print("----------")
            print("Waiting SQS message...")
            print("----------")
            response = self.queue.receive_message()

            if response:
                message = json.loads(response.body)

                server_region = message["StateMachine"].split(":")[3]

                self.session = TrainingSession(
                    task_name=message["TaskName"],
                    task_token=message["TaskToken"],
                    server_region=server_region,
                    direction=message["Direction"],
                    phase=message["Phase"],
                )

                if self.session.phase == "End":
                    self.session.s3_bucket = message["VFLBucket"]
                    self.__finalize()
                    self.__send_task_success()
                    response.delete()
                    print("End training.")
                    break

                self.session.batch_size = int(message["BatchSize"])
                self.session.batch_index = int(message["BatchIndex"])
                self.session.batch_count = int(message["BatchCount"])
                self.session.va_batch_index = int(message["VaBatchIndex"])
                self.session.va_batch_count = int(message["VaBatchCount"])
                self.session.is_next_batch = bool(message["IsNextBatch"])
                self.session.is_next_va_batch = bool(message["IsNextVaBatch"])
                self.session.epoch_index = int(message["EpochIndex"])
                self.session.is_next_epoch = bool(message["IsNextEpoch"])
                self.session.shuffled_index_path = message["ShuffledIndexPath"]

                print(f"Task Name: {self.session.task_name}")
                print(f"Phase: {self.session.phase}")
                print(f"Direction: {self.session.direction}")

                if self.session.phase == "Save":
                    print("Saving model...")
                    if not os.path.exists("model"):
                        os.mkdir("model")
                    model_name = f"model/{self.session.task_name}-client-model-{self.client_id}-best.pt"
                    self.model.save(model_name)
                else:
                    print(f"Epoch Count: {int(self.session.epoch_index) + 1}")
                    print(
                        f"Batch Count: {int(self.session.batch_index) + 1} / {self.session.batch_count}"
                    )
                    print(
                        f"Validation Batch Count: {int(self.session.va_batch_index) + 1} / {self.session.va_batch_count}"
                    )

                if self.session.phase == "Training":
                    if self.session.direction == "Forward":
                        self.session.s3_bucket = message["VFLBucket"]
                        self.embed_file = self.__forward(self.session)

                    elif self.session.direction == "Backward":
                        self.session.gradient_file_path = message["GradientFile"]
                        self.__backward()

                elif self.session.phase == "Validation":
                    self.session.s3_bucket = message["VFLBucket"]
                    self.embed_file = self.__validate()

                self.__send_task_success()
                response.delete()

    def __save_embed(self, bucket: str, key: str) -> str:
        file_name = f"{self.tmp_dir}/{key}"
        torch.save(self.embed.detach().cpu(), file_name)
        key = file_name.split("/")[-1]
        self.s3.meta.client.upload_file(file_name, bucket, key)
        return f"s3://{bucket}/{file_name}"

    def __set_gradient(self, s3_uri: str) -> None:
        bucket = s3_uri.split("/")[2]
        key = s3_uri.split("/")[-1]
        local_file_path = f"{self.tmp_dir}/{key}"
        self.s3.meta.client.download_file(bucket, key, local_file_path)
        self.gradient = torch.load(local_file_path).to()

    def __forward(self, session: TrainingSession) -> str:
        batch_index = session.batch_index
        batch_size = session.batch_size
        s3_bucket = session.s3_bucket
        head = batch_index * batch_size
        tail = min(head + batch_size, len(self.tr_uid))
        self.shuffled_index.update_if_not_set(self.session.shuffled_index_path)
        si = self.shuffled_index.index[head:tail]

        self.model.train()

        batch_x = self.tr_x[si, :].to()
        self.optimizer.zero_grad()
        self.embed = self.model(batch_x)

        key = f"{self.session.task_name}-tr-embed-{self.client_id}.pt"
        return self.__save_embed(s3_bucket, key)

    def __backward(self):
        self.model.train()
        self.__set_gradient(self.session.gradient_file_path)
        self.embed.backward(self.gradient)
        self.optimizer.step()

    def __validate(self):
        s3_bucket = self.session.s3_bucket
        va_batch_index = self.session.va_batch_index
        batch_size = self.session.batch_size
        head = batch_size * va_batch_index
        tail = min(head + batch_size, len(self.va_uid))

        self.model.eval()
        batch_x = self.va_x[head:tail, :].to()
        self.embed = self.model(batch_x)

        key = f"{self.session.task_name}-va-embed-{self.client_id}.pt"
        return self.__save_embed(s3_bucket, key)

    def __send_task_success(self):
        output = {}
        output["MemberId"] = str(self.client_id)

        if self.session.task_name:
            output["TaskName"] = self.session.task_name

        if self.session.phase == "End":
            output["TaskId"] = self.client_id.zfill(4) + "-end"
        else:
            output["TaskId"] = self.client_id.zfill(4) + str(
                self.session.batch_index
            ).zfill(8)
            output["SqsUrl"] = self.queue.sqs.url

        if self.session.direction is not None:
            output["Direction"] = self.session.direction

        if self.session.phase is not None:
            output["Phase"] = self.session.phase

        if self.session.batch_size is not None:
            output["BatchSize"] = self.session.batch_size

        if self.session.batch_index is not None:
            output["BatchIndex"] = self.session.batch_index

        if self.session.batch_count is not None:
            output["BatchCount"] = self.session.batch_count

        if self.session.va_batch_index is not None:
            output["VaBatchIndex"] = self.session.va_batch_index

        if self.session.va_batch_count is not None:
            output["VaBatchCount"] = self.session.va_batch_count

        if self.session.is_next_batch is not None:
            output["IsNextBatch"] = self.session.is_next_batch

        if self.session.is_next_va_batch is not None:
            output["IsNextVaBatch"] = self.session.is_next_va_batch

        if self.session.epoch_index is not None:
            output["EpochIndex"] = self.session.epoch_index

        if self.session.is_next_epoch is not None:
            output["IsNextEpoch"] = self.session.is_next_epoch

        if self.session.shuffled_index_path is not None:
            output["ShuffledIndexPath"] = self.session.shuffled_index_path

        if (
            self.session.phase == "Training" and self.session.direction == "Forward"
        ) or self.session.phase == "Validation":
            output["EmbedFile"] = self.embed_file

        output = json.dumps(output)
        stf_client = boto3.client(
            "stepfunctions", region_name=self.session.server_region
        )
        stf_client.send_task_success(
            taskToken=self.session.task_token,
            output=output,
        )
        stf_client.close()

    def __finalize(self):
        model_name = (
            f"model/{self.session.task_name}-client-model-{self.client_id}-best.pt"
        )
        key = model_name.split("/")[-1]
        self.s3.meta.client.upload_file(model_name, self.session.s3_bucket, key)
        os.remove(
            f"{self.tmp_dir}/{self.session.task_name}-tr-embed-{self.client_id}.pt"
        )
        os.remove(
            f"{self.tmp_dir}/{self.session.task_name}-va-embed-{self.client_id}.pt"
        )
        os.remove(
            f"{self.tmp_dir}/{self.session.task_name}-gradient-{self.client_id}.pt"
        )


if __name__ == "__main__":
    args = sys.argv
    if args[1] is None:
        sys.exit("Client is necessary to run")

    client = args[1]
    config_path = client_configs[client]
    with open(config_path, "r") as config_file:
        client_config = json.load(config_file)

    sqs_region = client_config["sqs_region"]
    sqs_name = client_config["sqs_name"]
    client_id = client_config["member_id"]

    dataset = Dataset(client_id)
    vfl_sqs = VFLSQS(sqs_name, sqs_region)
    model = ClientModel(len(dataset.tr_xcols), 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    shuffled_index = ShuffledIndex()
    client_trainer = ClientTrainer(
        client_id, vfl_sqs, dataset, model, optimizer, shuffled_index
    )
    client_trainer.start()
