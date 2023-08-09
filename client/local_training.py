import json
import sys
import tempfile
from urllib.parse import urlparse
import numpy as np
import torch
import torch.nn.functional as F
import math
import random
import boto3
import os

client_configs = {
    "1": "config/1.json",
    "2": "config/2.json",
    "3": "config/3.json",
    "4": "config/4.json",
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class S3Url:
    def __init__(self, url) -> None:
        self.url = url
        parsed_url = urlparse(url, allow_fragments=False)
        self.bucket = parsed_url.netloc
        self.key = parsed_url.path.lstrip("/")
        self.file_name = self.key.split("/")[-1]
        self.prefix = self.key[: -len(self.file_name)]

class Encoder:
    def __init__(self,encode_type)-> None:
        self.encode_type = encode_type
        
    def encode(self,src_info):
        if self.encode_type == 'emb_svfl16':
            ec, pos = self.sparse_encode16(src_info['embed'], get_position=2)
            result = {
                'ec' : ec,
                'pos' : pos,
            }
            return result
        if self.encode_type == 'emb':
            result = {
            'ec' : src_info['embed'],
            }
            return result
        if self.encode_type == 'gradient_svfl16':
            gec = self.sparse_encode16(src_info['ge'], get_position=1, nz_pos=src_info['nz_pos'])
            result = {
            'gec': gec,
            }
            return result
        if self.encode_type == 'gradient':
            result = {
                'gec' : src_info['ge'],
            }
            return result
        
    def sparse_encode16(self, src, get_position=0, nz_pos=None, axis=1):
        # src (gpu, float32), nz_pos (cpu, bool)
        samples = src.shape[0]
        dims    = src.shape[1]

        # run-length compress by axis=1
        if axis==1:
            dst = src.detach().cpu().t().reshape(-1)
        else:
            dst = src.detach().cpu().reshape(-1)
        length = len(dst)

        if nz_pos is None:
            nz_pos = (dst!=0)

        non_zero_values = dst[nz_pos].to(torch.float16)

        #######################
        # Extract change points
        #######################
        nz_pos = nz_pos.char()
        ind = torch.arange(0, length)#, dtype=torch.int16)

        # none zero position
        nz_cp = nz_pos - torch.cat((torch.CharTensor([0]), nz_pos[0:-1]), 0)
        nz_head = ind[nz_cp==1]
        nz_tail = ind[nz_cp==-1]

        if get_position==0:
            # Fully encode
            info = {
                'samples': samples, #int
                'dims': dims, #int
                'non_zero_values': non_zero_values, # float16, tensor, 1d-array
                'nz_head': nz_head.to(torch.int16), # int16, tensor, 1d-array
                'nz_tail': nz_tail.to(torch.int16), # int16, tensor, 1d-array
            }
            return info
        elif get_position==1:
            # Without position
            info = {
                'samples': samples, #int
                'dims': dims, #int
                'non_zero_values': non_zero_values, # float16, tensor, 1d-array
            }
            return info
        elif get_position==2:
            # Duplicate position
            info = {
                'samples': samples, #int
                'dims': dims, #int
                'non_zero_values': non_zero_values, # float16, tensor, 1d-array
                'nz_head': nz_head.to(torch.int16), # int16, tensor, 1d-array
                'nz_tail': nz_tail.to(torch.int16), # int16, tensor, 1d-array
            }
            position = {
                'nz_head': nz_head, # long, tensor, cpu, 1d-array
                'nz_tail': nz_tail, # long, tensor, cpu, 1d-array
            }
            return info, position
        
class Decoder:
    def __init__(self, decode_type):
        self.decode_type = decode_type
        
    def decode(self,src_info):
        result = {}
        if self.decode_type == 'gradient_svfl16':
            gradient,_ = self.sparse_decode16(src_info['ge'], set_position=src_info['pos'])
            return gradient
        if self.decode_type == 'gradient':
            return src_info['ge']
        if self.decode_type == 'emb_svfl16':
            value, nz_pos = self.sparse_decode16(info = src_info['e'])
            result = {
                'value': value,
                'nz_pos':nz_pos,
            }
            return result
        if self.decode_type == 'emb':
            result = {
            'value': src_info['e'],
            }
            return result
        
    def sparse_decode16(self,info, set_position=None):
        # Extract data
        samples = info['samples']
        dims    = info['dims']
        length  = samples * dims # info['length']
        if set_position is None:
            nz_head = info['nz_head'].long()
            nz_tail = info['nz_tail'].long()
        else:
            nz_head = set_position['nz_head']#.long()
            nz_tail = set_position['nz_tail']#.long()

        nz_cp = torch.zeros(length, dtype=torch.int8)

        nz_cp[nz_head] = 1
        nz_cp[nz_tail] = -1

        nz_pos = torch.cumsum(nz_cp, dim=0).bool()

        dst = torch.zeros(length)
        dst[nz_pos] = info['non_zero_values'].float()

        dst = dst.reshape((dims, samples)).t().to()

        return dst, nz_pos
        
class Dataset:
    def __init__(self, client) -> None:
        self.tr_uid = torch.LongTensor(
            np.load(f"dataset/client{client}/tr_uid.npy", allow_pickle=False)
        )
        self.tr_x = torch.FloatTensor(
            np.load(f"dataset/client{client}/tr_x.npy", allow_pickle=False)
        )
        self.tr_xcols = np.load(
            f"dataset/client{client}/cols.npy", allow_pickle=False
        ).tolist()
        self.va_uid = torch.LongTensor(
            np.load(f"dataset/client{client}/va_uid.npy", allow_pickle=False)
        )
        self.va_x = torch.FloatTensor(
            np.load(f"dataset/client{client}/va_x.npy", allow_pickle=False)
        )
        self.va_xcols = self.tr_xcols
        self.tr_sample_count = len(self.tr_uid)
        self.va_sample_count = len(self.va_uid)
        

        
        






class Gradient:
    def __init__(self, s3_object,decoder,src_info) -> None:
        self.s3_object = s3_object

        with tempfile.TemporaryDirectory() as tmpdirname:
            file_path = f'{tmpdirname}/ge.pt'
            s3_object.download_file(file_path)
            src_info['ge'] = torch.load(file_path)
            ge = decoder.decode(src_info)
            self.value = ge

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
        self.i2h = torch.nn.Linear(in_size, hidden_size, bias=False)
        # Reproducability
        set_seed(seed)
        torch.nn.init.xavier_uniform_(self.i2h.weight.data)
            
                
    def forward(self, x):
        h = self.i2h(x)
        h = F.relu(h)
        return h
    
    def save(self, file_path: str):
        torch.save(self.state_dict(), file_path)

class VFLSQS:
    def __init__(self, name, region) -> None:
        self.name = name
        self.region = region

        client = boto3.client("sqs", region_name=region)
        self.url = client.get_queue_url(QueueName=self.name)["QueueUrl"]
        self.sqs = boto3.resource("sqs", region_name=region).Queue(self.url)

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
            file_name = uri.split("/")[-1]
            bucket = boto3.resource("s3").Bucket(bucket_name)
            with tempfile.TemporaryDirectory() as tmpdirname:
                bucket.download_file(key, f"{tmpdirname}/{file_name}")
                self.index = torch.LongTensor(
                    np.load(f"{tmpdirname}/{file_name}", allow_pickle=False)
                )
        else:
            self.index = torch.LongTensor(np.load(uri, allow_pickle=False))

    def update_if_not_set(self, uri: str):
        if self.index is None:
            self.update(uri=uri)
        

class ClientTrainer:
    def __init__(
        self,
        client_id: str,
        dataset: Dataset,
        model: ClientModel,
        optimizer: torch.optim.Adam,
        shuffled_index: ShuffledIndex,
    ) -> None:
        self.client_id = client_id
        self.tr_uid = dataset.tr_uid
        self.tr_x = dataset.tr_x
        self.tr_xcols = dataset.tr_xcols
        self.va_uid = dataset.va_uid
        self.va_x = dataset.va_x
        self.va_xcols = dataset.va_xcols

        self.shuffled_index = shuffled_index
        
        self.model = model.to()
        self.best_model = model.to()
        self.optimizer = optimizer
        self.embed: torch.FloatTensor = None
        self.va_embed: torch.FloatTensor = None
        self.encoded_result = None 
        self.va_encoded_result = None

    def save_embed(self, s3_object,encoder) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = f"{tmpdirname}/_emb.pt"
            src_info = {}
            src_info['embed'] = self.embed.detach().cpu()
            self.encoded_result = encoder.encode(src_info)
            torch.save(self.encoded_result['ec'],path)
            s3_object.upload_file(path)
           
   
    def save_va_embed(self, s3_object,encoder) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = f"{tmpdirname}/va_emb.pt"
            src_info = {}
            src_info['embed'] = self.va_embed
            self.va_encoded_result = encoder.encode(src_info)
            torch.save(self.va_encoded_result['ec'],path)
            s3_object.upload_file(path)

    def forward(self, batch_size, batch_index) -> str:
        head = batch_index * batch_size
        tail = min(head + batch_size, len(self.tr_uid))
        si = self.shuffled_index.index[head:tail]

        self.model.train()

        batch_x = self.tr_x[si, :].to()
        self.optimizer.zero_grad()
        self.embed = self.model(batch_x)

    def backward(self, gradient: Gradient) -> None:
        self.model.train()
        self.embed.backward(gradient.value)
        self.optimizer.step()

    def validate(self, batch_size: int, va_batch_index: int) -> None:
        head = batch_size * va_batch_index
        tail = min(head + batch_size, len(self.va_uid))

        self.model.eval()
        batch_x = self.va_x[head:tail, :].to()
        self.va_embed = self.model(batch_x)

    def commit_model(self):
        self.best_model = self.model

    def save_model(self, s3_object):
        with tempfile.TemporaryDirectory() as tmpdirname:
            path = f"{tmpdirname}/client-model.pt"
            torch.save(self.best_model.state_dict(), path)
            s3_object.upload_file(path)


def send_task_success(client, token: str, output: dict):
    str_output = json.dumps(output)
    client.send_task_success(
        taskToken=token,
        output=str_output,
    )


if __name__ == "__main__":
    args = sys.argv
    if args[1] is None:
        sys.exit("Client is necessary to run")

    client = args[1]
    config_path = client_configs[client]
    with open(config_path, "r") as config_file:
        client_config = json.load(config_file)

    iam_user_id = boto3.client("sts").get_caller_identity()["UserId"]
    s3 = boto3.resource("s3")
    stf_client = None

    sqs_region = client_config["sqs_region"]
    sqs_name = client_config["sqs_name"]
    client_id = client_config["member_id"]

    dataset = Dataset(client_id)
    vfl_sqs = VFLSQS(sqs_name, sqs_region)
    model = ClientModel(len(dataset.tr_xcols), 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    shuffled_index = ShuffledIndex()
    client_trainer = ClientTrainer(
        client_id,
        dataset,
        model,
        optimizer,
        shuffled_index,
    )
    while True:
        print("----------")
        print("Waiting SQS message...")
        print("----------")
        response = vfl_sqs.receive_message()

        if response:
            message = json.loads(response.body)
            output = message
            output["MemberId"] = str(client_id)
            task_name = message["TaskName"]
            task_token = message["TaskToken"]
            server_region = message["StateMachine"].split(":")[3]
            phase = message["Phase"]
            s3_bucket = message["VFLBucket"]
            

            if stf_client is None:
                stf_client = boto3.client(
                    "stepfunctions",
                    region_name=server_region,
                )
            try:
                batch_index = int(message["BatchIndex"])
            except:
                batch_index = -1
            
            
            s3_best_model_object = s3.Object(
                s3_bucket,
                f"model/{task_name}-client-model-{client_id}-best.pt",
            )
            s3_tr_embed_object = s3.Object(
                s3_bucket, f"{iam_user_id}/{task_name}-tr-embed-{str(batch_index)}-{client_id}.npy"
            )
            s3_va_embed_object = s3.Object(
                s3_bucket, f"{iam_user_id}/{task_name}-va-embed-{client_id}.npy"
            )

            if phase == "End":
                client_trainer.save_model(s3_object=s3_best_model_object)
                output["TaskId"] = client_id.zfill(4) + "-end"
                send_task_success(
                    client=stf_client,
                    token=task_token,
                    output=output,
                )
                response.delete()
                print("End training.")
                break
                
            Use_Sparse = message['UseSparse']
            direction = message["Direction"]
            batch_size = int(message["BatchSize"])
            batch_index = int(message["BatchIndex"])
            batch_count = int(message["BatchCount"])
            va_batch_index = int(message["VaBatchIndex"])
            va_batch_count = int(message["VaBatchCount"])
            is_next_batch = bool(message["IsNextBatch"])
            is_next_va_batch = bool(message["IsNextVaBatch"])
            epoch_index = int(message["EpochIndex"])
            is_next_epoch = bool(message["IsNextEpoch"])
            shuffled_index_path = message["ShuffledIndexPath"]

            output["TaskId"] = client_id.zfill(4) + str(batch_index).zfill(8)
            output["SqsUrl"] = vfl_sqs.url
            

            print(f"Task Name: {task_name}")
            print(f"Phase: {phase}")
            print(f"Direction: {direction}")
            # Define encoder and decoder for embedding and gradient
            if Use_Sparse == True:
                print('Use Sparse')
                emb_encoder = Encoder('emb_svfl16')
                gradient_decoder = Decoder('gradient_svfl16')
            else:
                print('Regular')
                emb_encoder = Encoder('emb')
                gradient_decoder = Decoder('gradient')
                
            if phase == "Save":
                print("Saving model...")
                client_trainer.commit_model()
            else:
                print(f"Epoch Count: {int(epoch_index) + 1}")
                print(f"Batch Count: {int(batch_index) + 1} / {batch_count}")
                print(
                    f"Validation Batch Count: {int(va_batch_index) + 1} / {va_batch_count}"
                )

            if phase == "Training":
                if direction == "Forward":
                    client_trainer.shuffled_index.update_if_not_set(shuffled_index_path)
                    client_trainer.forward(
                        batch_index=batch_index, batch_size=batch_size
                    )
                    client_trainer.save_embed(s3_tr_embed_object,encoder = emb_encoder)
                    output[
                        "EmbedFile"
                    ] = f"s3://{s3_tr_embed_object.bucket_name}/{s3_tr_embed_object.key}"
                elif direction == "Backward":
                    gradient_url = S3Url(url=message["GradientFile"])
                    s3_gradient_object = s3.Object(
                        gradient_url.bucket, gradient_url.key
                    )
                    gradient = Gradient(s3_object=s3_gradient_object, decoder = gradient_decoder, src_info =  client_trainer.encoded_result)
                    client_trainer.backward(gradient=gradient)

            elif phase == "Validation":
                client_trainer.validate(
                    batch_size=batch_size, va_batch_index=va_batch_index
                )
                client_trainer.save_va_embed(s3_object=s3_va_embed_object, encoder = emb_encoder)
                output[
                    "EmbedFile"
                ] = f"s3://{s3_va_embed_object.bucket_name}/{s3_va_embed_object.key}"
            send_task_success(
                client=stf_client,
                token=task_token,
                output=output,
            )
            response.delete()