import numpy as np
import torch
import random
import boto3
import json
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

seed = 42 # Random Seed
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class ServerModel(torch.nn.Module):
    def __init__(self, hidden_size, out_size):
        super(ServerModel, self).__init__()
        self.h2h = torch.nn.Linear(hidden_size, hidden_size//2)
        self.h2o = torch.nn.Linear(hidden_size//2, out_size)
        
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

class TrainingSession():
    def __init__(self, task_name, num_of_clients, epoch_index, batch_index, batch_size, va_batch_index) -> None:
        self.task_name = task_name
        self.num_of_clients = num_of_clients
        self.epoch_index = epoch_index
        self.batch_index = batch_index
        self.batch_size = batch_size
        self.va_batch_index = va_batch_index

class ServerTrainer():
    def __init__(self, training_session, s3_bucket) -> None:
        self.tmp_dir = "/tmp/"
        self.task_name = training_session.task_name
        self.num_of_clients = training_session.num_of_clients
        self.client_ids = [ str(i+1) for i in range(self.num_of_clients) ]
        self.epoch_index = training_session.epoch_index
        self.batch_index = training_session.batch_index
        self.batch_size = training_session.batch_size
        self.va_batch_index = training_session.va_batch_index
        self.s3_bucket = s3_bucket
        self.embeds = dict()
        self.gradients = dict()

        # Init tr_y
        self.tr_y = torch.load('tr_y.pt')

        # Init pos_weight
        self.pos_weight = (self.tr_y.shape[0] - self.tr_y.sum()) / self.tr_y.sum()

        # Init criterion
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        # Init shuffled index
        shuffled_index_path = self.__download_file_from_s3(f"{self.task_name}-shuffled-index.pt")
        self.shuffled_index = torch.load(shuffled_index_path)

        # Init tr_uid
        self.tr_uid = torch.load('tr_uid.pt')

        # Init va_uid
        self.va_uid = torch.load('va_uid.pt')

        # Init va_y
        self.va_y = torch.load('va_y.pt')

        # Init server model
        self.server_model = ServerModel(4*self.num_of_clients, 1)
        if self.epoch_index != 0 or self.batch_index != 0:
            server_model_file = self.__download_file_from_s3(f"{self.task_name}-server-model.pt")
            self.server_model.load_state_dict(torch.load(server_model_file))

        # Init optimizer
        self.optimizer = torch.optim.Adam(self.server_model.parameters(), lr=0.01)
        if self.epoch_index != 0 or self.batch_index != 0:
            optimizer_file = self.__download_file_from_s3(f"{self.task_name}-optimizer.pt")
            self.optimizer.load_state_dict(torch.load(optimizer_file))

        # Load total Loss
        if self.batch_index == 0:
            self.total_tr_loss = 0
            self.total_va_loss = 0
        else:
            loss_file_path = self.__download_file_from_s3(f"{self.task_name}-loss.json")
            with open(loss_file_path) as f:
                loss = json.load(f)
                self.total_tr_loss = loss["total_tr_loss"]
                self.total_va_loss = loss["total_va_loss"]

        # Load pred and true for training
        self.tr_true = self.tr_y[self.shuffled_index, :]
        if self.batch_index == 0:
            self.tr_pred = np.zeros(self.tr_y.shape)
        else:
            tr_pred_file = self.__download_file_from_s3(f"{self.task_name}-tr-pred.pt")
            self.tr_pred = torch.load(tr_pred_file)

        # Load pred and true for validation
        self.va_true = self.va_y
        if self.va_batch_index == 0:
            self.va_pred = np.zeros(self.va_y.shape)
        else:
            va_pred_file = self.__download_file_from_s3(f"{self.task_name}-va-pred.pt")
            self.va_pred = torch.load(va_pred_file)

    def __download_file_from_s3(self, s3_key) -> str:
        file_name = s3_key.split("/")[-1]
        download_path = self.tmp_dir + file_name
        s3 = boto3.resource('s3')
        s3.meta.client.download_file(self.s3_bucket, s3_key, download_path)
        return download_path

    def __upload_file_to_s3(self, file_path, s3_key) -> bool:
        client = boto3.client('s3')
        client.upload_file(file_path, self.s3_bucket, s3_key)
        return True

    def set_embed(self, client_id, s3_key) -> None:
        file_name = s3_key.split("/")[-1]
        file_path = f'/tmp/{file_name}'
        self.__download_file_from_s3(file_name)
        self.embeds[client_id] = torch.load(file_path)

    def save_gradient(self, file_name_prefix=None) -> dict:
        embed_files_s3_path = dict()
        for client_id in self.gradients.keys():
            file_name = f"{self.task_name if file_name_prefix is None else file_name_prefix}-gradient-{client_id}.pt"
            local_path = self.tmp_dir + file_name
            torch.save(self.gradients[client_id], local_path)
            self.__upload_file_to_s3(local_path, file_name)
            embed_files_s3_path[client_id] = f"s3://{self.s3_bucket}/{file_name}"
        return embed_files_s3_path

    def save_model(self, file_name=None) -> None:
        file_name = f"{self.task_name}-server-model.pt" if file_name is None else file_name
        model_file_path = self.tmp_dir + file_name
        if file_name:
            model_file_path = f"/tmp/{file_name}"
        key = model_file_path.split("/")[-1]
        torch.save(self.server_model.state_dict(), model_file_path)
        self.__upload_file_to_s3(model_file_path, key)

    def save_optimizer(self, file_name=None) -> None:
        file_name = f"{self.task_name}-optimizer.pt" if file_name is None else file_name
        optimizer_path = self.tmp_dir + file_name
        torch.save(self.optimizer.state_dict(), optimizer_path)
        self.__upload_file_to_s3(optimizer_path, file_name)

    def save_loss(self, file_name=None) -> None:
        file_name = f"{self.task_name}-loss.json" if file_name is None else file_name
        loss_file = self.tmp_dir + file_name
        loss = {
            "total_tr_loss": self.total_tr_loss,
            "total_va_loss": self.total_va_loss,
        }
        with open(loss_file, 'w') as f:
            json.dump(loss, f)
        self.__upload_file_to_s3(loss_file, file_name)

    def save_tr_pred(self, file_name=None) -> None:
        file_name = f"{self.task_name}-tr-pred.pt" if file_name is None else file_name
        local_file_path = self.tmp_dir + file_name
        torch.save(self.tr_pred, local_file_path)
        self.__upload_file_to_s3(local_file_path, file_name)

    def save_va_pred(self, file_name=None) -> None:
        file_name = f"{self.task_name}-va-pred.pt" if file_name is None else file_name
        local_file_path = self.tmp_dir + file_name
        torch.save(self.va_pred, local_file_path)
        self.__upload_file_to_s3(local_file_path, file_name)

    def train(self) -> None:
        tr_sample_count = len(self.tr_uid)
        head = self.batch_index * self.batch_size
        tail = min(head + self.batch_size, tr_sample_count)
        si = self.shuffled_index[head:tail]

        self.server_model.train()
        batch_y = self.tr_y[si, :]
        self.optimizer.zero_grad()
        embed_tuple = ()
        for client_id in self.client_ids:
            embed_tuple = (*embed_tuple, self.embeds[client_id])
        embed = torch.cat(embed_tuple, 1)
        embed.requires_grad_(True)

        pred_y = self.server_model(embed)
        loss = self.criterion(pred_y, batch_y)
        loss.backward()
        self.optimizer.step()
        self.total_tr_loss += loss.item()

        for i, client_id in enumerate(self.client_ids):
            e_head = i * 4
            e_tail = (i+1) * 4
            self.gradients[client_id] = embed.grad[:,e_head:e_tail].cpu()

        self.tr_pred[head:tail,:] = torch.sigmoid(pred_y).detach().cpu().numpy()

    def validate(self) -> None:
        va_sample_count = len(self.va_uid)
        head = self.va_batch_index * self.batch_size
        tail = min(head + self.batch_size, va_sample_count)

        self.server_model.eval()
        batch_y = self.va_y[head:tail, :]
        embed_tuple = ()
        for client_id in self.client_ids:
            embed_tuple = (*embed_tuple, self.embeds[client_id])
        embed = torch.cat(embed_tuple, 1)

        pred_y = self.server_model(embed)
        loss = self.criterion(pred_y, batch_y)
        self.total_va_loss += loss.item()
        self.va_pred[head:tail, :] = torch.sigmoid(pred_y).detach().cpu().numpy()

    def get_tr_loss(self) -> float:
        return self.total_tr_loss / (self.batch_index + 1)

    def get_tr_auc(self) -> float:
        return roc_auc_score(self.tr_true, self.tr_pred)

    def get_va_loss(self) -> float:
        return self.total_va_loss / (self.va_batch_index + 1)

    def get_va_auc(self) -> float:
        return roc_auc_score(self.va_true, self.va_pred)

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

    # Init Server Trainer
    session = TrainingSession(
        task_name=task_name,
        num_of_clients=num_of_clients,
        epoch_index=epoch_index,
        batch_index=batch_index,
        batch_size=batch_size,
        va_batch_index=va_batch_index,
    )
    server_trainer = ServerTrainer(training_session=session, s3_bucket=s3_bucket)

    # Save model and return
    if phase == "Save":
        # Save best model
        best_model_name = f"{task_name}-server-model-best.pt"
        server_trainer.save_model(file_name=best_model_name)

        # Generate response for the next Map step
        response = []
        for input_item in input_items:
            queue_url = input_item["SqsUrl"]
            response.append({
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
            })

        print(json.dumps(response))
        return response

    # Set embed from clients
    for input_item in input_items:
        client_id = input_item["MemberId"]
        embed_file = input_item["EmbedFile"].split("/")[-1]
        server_trainer.set_embed(client_id, embed_file)

    response = None

    # Training
    if phase == "Training":
        # Train model
        server_trainer.train()

        # Save parameters
        server_trainer.save_loss()
        server_trainer.save_tr_pred()
        gradient_files = server_trainer.save_gradient()

        # Save model and optimizer
        server_trainer.save_model()
        server_trainer.save_optimizer()

        # Generate response for the next Map step
        response = []
        for input_item in input_items:
            queue_url = input_item["SqsUrl"]
            member_id = input_item["MemberId"]
            response.append({
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
            })
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
            response.append({
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
            })

    print(json.dumps(response))
    return response