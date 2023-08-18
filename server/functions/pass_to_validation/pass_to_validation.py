def lambda_handler(event, context):
    print(event)
    response = []
    for item in event["InputItems"]:
        task_name = item["TaskName"]
        batch_index = int(item["BatchIndex"])
        batch_count = int(item["BatchCount"])
        batch_size = int(item["BatchSize"])
        epoch_index = int(item["EpochIndex"])
        epoch_count = int(item["EpochCount"])
        patience = int(item["Patience"])
        sparse_encoding = bool(item["SparseEncoding"])
        sparse_lambda = float(item["SparseLambda"])
        va_batch_index = 0
        va_batch_count = int(item["VaBatchCount"])
        shuffled_index_path = item["ShuffledIndexPath"]
        is_next_batch = bool(item["IsNextBatch"])
        is_next_va_batch = 0 + 1 < va_batch_count
        is_next_epoch = bool(item["IsNextEpoch"])
        queue_url = item["SqsUrl"]
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
                "Patience": patience,
                "SparseEncoding": sparse_encoding,
                "SparseLambda": sparse_lambda,
                "IsNextEpoch": is_next_epoch,
                "ShuffledIndexPath": shuffled_index_path,
                "SqsUrl": queue_url,
                "VFLBucket": vfl_bucket,
            }
        )

    return response
