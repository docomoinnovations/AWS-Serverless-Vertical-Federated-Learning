def lambda_handler(event, context):
    print(event)

    response = []
    for item in event["InputItems"]:
        task_name = item["TaskName"]
        batch_index = int(item["BatchIndex"])
        batch_count = int(item["BatchCount"])
        batch_size = int(item["BatchSize"])
        va_batch_index = int(item["VaBatchIndex"]) + 1
        va_batch_count = int(item["VaBatchCount"])
        patience = int(item["Patience"])
        shuffled_index_path = item["ShuffledIndexPath"]
        sparse_encoding = item["SparseEncoding"]
        sparse_lambda = float(item["SparseLambda"])
        is_next_batch = bool(item["IsNextBatch"])
        is_next_va_batch = va_batch_index + 1 < va_batch_count
        epoch_index = int(item["EpochIndex"])
        epoch_count = int(item["EpochCount"])
        is_next_epoch = bool(item["IsNextEpoch"])
        sqs_url = item["SqsUrl"]
        vfl_bucket = item["VFLBucket"]

        response.append(
            {
                "TaskName": task_name,
                "BatchIndex": batch_index,
                "BatchCount": batch_count,
                "BatchSize": batch_size,
                "VaBatchIndex": va_batch_index,
                "VaBatchCount": va_batch_count,
                "IsNextBatch": is_next_batch,
                "IsNextVaBatch": is_next_va_batch,
                "EpochIndex": epoch_index,
                "EpochCount": epoch_count,
                "IsNextEpoch": is_next_epoch,
                "Patience": patience,
                "SparseEncoding": sparse_encoding,
                "SparseLambda": sparse_lambda,
                "ShuffledIndexPath": shuffled_index_path,
                "SqsUrl": sqs_url,
                "VFLBucket": vfl_bucket,
            }
        )

    return response
