import boto3
import json

class Score():
  def __init__(self, best_score, patience_counter) -> None:
    self.best_score = best_score
    self.patience_counter = patience_counter
    
def get_score(s3_bucket, task_name):
    file_name = f"{task_name}-score.json"
    local_file_path = f"/tmp/{file_name}"
    s3 = boto3.resource('s3')
    s3.meta.client.download_file(s3_bucket, file_name, local_file_path)
    with open(local_file_path, 'r') as f:
        score_data = json.load(f)
        best_score = score_data["best_score"]
        patience_counter = score_data["patience_counter"]
        return Score(best_score, patience_counter)

def save_score(score, s3_bucket, task_name):
    file_name = f"{task_name}-score.json"
    local_file_path = f"/tmp/{file_name}"
    score_data = {
        "best_score": score.best_score,
        "patience_counter": score.patience_counter
    }
    with open(local_file_path, 'w') as f:
        json.dump(score_data, f)
    
    client = boto3.client('s3')
    client.upload_file(local_file_path, s3_bucket, file_name)

def lambda_handler(event, context):
    print(event)

    s3_bucket = event["VFLBucket"]
    patience = event["Patience"]
    items = event["InputItems"]

    task_name = items[0]["TaskName"]
    batch_index = items[0]["BatchIndex"]
    va_batch_index = items[0]["VaBatchIndex"]
    batch_count = int(items[0]["BatchCount"])
    va_batch_count = int(items[0]["VaBatchCount"])
    shuffled_index_path = items[0]["ShuffledIndexPath"]
    epoch_index = int(items[0]["EpochIndex"])
    is_next_batch = bool(items[0]["IsNextBatch"])
    is_next_va_batch = bool(items[0]["IsNextVaBatch"])
    is_next_epoch = bool(items[0]["IsNextEpoch"])
    va_auc = float(items[0]["VaAuc"])

    is_best_score = False
    current_score = Score(best_score=-1, patience_counter=0) if epoch_index == 0 else get_score(s3_bucket=s3_bucket, task_name=task_name)

    if va_auc > current_score.best_score:
        current_score.best_score = va_auc
        is_best_score = True
        current_score.patience_counter = 0

    else:
        current_score.patience_counter += 1
        if current_score.patience_counter > patience:
            is_next_epoch = False
    
    save_score(score=current_score, s3_bucket=s3_bucket, task_name=task_name)

    response = []
    for item in items:
        sqs_url = item["SqsUrl"]
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
            "IsBestScore": is_best_score,
            "SqsUrl": sqs_url,
        })

    print(json.dumps(response))
    return response