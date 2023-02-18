def lambda_handler(event, context):
  print(event)
  epoch_count = event["EpochCount"]
  response = []
  for item in event["InputItems"]:
    task_name = item["TaskName"]
    batch_index = 0
    va_batch_index = 0
    # va_batch_index = int(item["VaBatchIndex"])
    batch_count = int(item["BatchCount"])
    va_batch_count = int(item["VaBatchCount"])
    shuffled_index_path = item["ShuffledIndexPath"]
    epoch_index = int(item["EpochIndex"]) + 1
    sqs_url = item["SqsUrl"]

    is_next_batch = batch_index + 1 < batch_count
    is_next_va_batch = va_batch_index + 1 < va_batch_count
    is_next_epoch = epoch_index + 1 < epoch_count

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
      "SqsUrl": sqs_url,
    })

  return response