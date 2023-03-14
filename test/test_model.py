import math
import os
import boto3
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import roc_auc_score


def get_training_attr(stack_name):
    cloudformation = boto3.resource("cloudformation")
    stack = cloudformation.Stack(stack_name)

    state_machine_arn = ""
    for output in stack.outputs:
        if output["OutputKey"] == "StateMachineArn":
            state_machine_arn = output["OutputValue"]
            break

    sfn_client = boto3.client("stepfunctions")
    execution_arn = sfn_client.list_executions(
        stateMachineArn=state_machine_arn,
        statusFilter="SUCCEEDED",
    )["executions"][0]["executionArn"]

    execution = sfn_client.describe_execution(executionArn=execution_arn)
    start_date = execution["startDate"]
    stop_date = execution["stopDate"]
    total_time = (stop_date - start_date).total_seconds()

    execution_output = json.loads(execution["output"])

    task_name = execution_output["Payload"]["TaskName"]
    num_of_clients = execution_output["Payload"]["NumOfClients"]
    epoch_count = execution_output["Payload"]["EpochCount"]
    batch_size = execution_output["Payload"]["BatchSize"]
    patience = execution_output["Payload"]["Patience"]
    s3_bucket = execution_output["Payload"]["VFLBucket"]

    return {
        "TaskName": task_name,
        "TotalTime": total_time,
        "NumOfClients": num_of_clients,
        "EpochCount": epoch_count,
        "BatchSize": batch_size,
        "Patience": patience,
        "StateMachineArn": state_machine_arn,
        "ExecutionArn": execution_arn,
        "S3Bucket": s3_bucket,
    }


def download_models(s3_bucket, task_name, num_of_clients):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket)
    server_model = f"{task_name}-server-model-best.pt"
    bucket.download_file(server_model, server_model)
    for i in range(num_of_clients):
        client_model = f"{task_name}-client-model-{i+1}-best.pt"
        bucket.download_file(client_model, client_model)


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
        h = self.i2h(x)
        h = F.relu(h)
        return h


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


argparser = ArgumentParser()
argparser.add_argument(
    "-s", "--stack-name", type=str, help="Specify CloudFormation stack name of VFL"
)


args = argparser.parse_args()
stack_name = args.stack_name
training_attr = get_training_attr(stack_name)
task_name = training_attr["TaskName"]
epoch_count = training_attr["EpochCount"]
batch_size = training_attr["BatchSize"]
num_of_clients = training_attr["NumOfClients"]
patience = training_attr["Patience"]
total_time = training_attr["TotalTime"]
state_machine_arn = training_attr["StateMachineArn"]
execution_arn = training_attr["ExecutionArn"]
s3_bucket = training_attr["S3Bucket"]

download_models(s3_bucket, task_name, num_of_clients)

###################
# Common
###################
te_uid = torch.LongTensor(np.load("te_uid.npy", allow_pickle=False))
te_sample_count = len(te_uid)
te_batch_count = math.ceil(te_sample_count / batch_size)

###################
# Client
###################
client_models = []
te_x_data = []
for i in range(num_of_clients):
    xcols = np.load(f"cols_{i+1}.npy", allow_pickle=False).tolist()
    client_model = ClientModel(len(xcols), 4)
    client_model.load_state_dict(torch.load(f"{task_name}-client-model-{i+1}-best.pt"))
    client_model.eval()
    client_models.append(client_model)
    te_x = torch.FloatTensor(np.load(f"te_x_{i+1}.npy", allow_pickle=False))
    te_x_data.append(te_x)

###################
# Server
###################
te_y = torch.FloatTensor(np.load("te_y.npy", allow_pickle=False))
server_model = ServerModel(4 * num_of_clients, 1)
server_model.load_state_dict(torch.load(f"{task_name}-server-model-best.pt"))
server_model.eval()
te_loss = 0
te_pred = np.zeros(te_y.shape)
te_true = np.zeros(te_y.shape)

for batch_index in tqdm(range(te_batch_count)):
    ###################
    # Common
    ###################
    head = batch_index * batch_size
    tail = min(head + batch_size, te_sample_count)
    batch_uid = te_uid[head:tail]

    ###################
    # Client
    ###################
    for i, client_model in enumerate(client_models):
        batch_x = te_x_data[i][head:tail, :]
        client_embed = client_model(batch_x)
        np.save(
            f"te_embed_{i+1}.npy",
            client_embed.detach().cpu().numpy(),
            allow_pickle=False,
        )

    ###################
    # Server: Receive
    ###################
    batch_y = te_y[head:tail, :]
    embed = ()
    for i in range(num_of_clients):
        e = torch.FloatTensor(np.load(f"te_embed_{i+1}.npy", allow_pickle=False))
        embed = (*embed, e)
    embed = torch.cat(embed, 1)
    pred_y = server_model(embed)

    te_pred[head:tail, :] = torch.sigmoid(pred_y).detach().cpu().numpy()
    te_true[head:tail, :] = batch_y.detach().cpu().numpy()

te_auc = roc_auc_score(te_true, te_pred)

print(f"ROC-AUC:           {te_auc:.4f}")
print(f"Task Name:         {task_name}")
print(f"Total time:        {total_time:,}s")
print(f"Number of clients: {num_of_clients:,}")
print(f"Batch Size:        {batch_size:,}")
print(f"Epoch:             {epoch_count:,}")
print(f"Patience:          {patience:,}")
print(f"State Machine ARN: {state_machine_arn}")
print(f"Execution ARN:     {execution_arn}")

###################
# Clean up
###################
os.remove(f"{task_name}-server-model-best.pt")
for i in range(num_of_clients):
    os.remove(f"te_embed_{i+1}.npy")
    os.remove(f"{task_name}-client-model-{i+1}-best.pt")
