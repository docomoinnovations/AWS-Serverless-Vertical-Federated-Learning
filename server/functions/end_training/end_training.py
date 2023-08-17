def lambda_handler(event, context):
    return {
        "TaskName": event[0]["TaskName"],
        "NumOfClients": len(event),
        "BatchSize": event[0]["BatchSize"],
        "EpochCount": event[0]["EpochCount"],
        "VFLBucket": event[0]["VFLBucket"],
        "SparseEncoding": event[0]["SparseEncoding"],
        "SparseLambda": event[0]["SparseLambda"],
    }
