def lambda_handler(event, context):
    vfl_bucket = event["VFLBucket"]
    input_items = event["InputItems"]
    num_of_clients = event["NumOfClients"]
    batch_size = event["BatchSize"]
    epoch_count = event["EpochCount"]
    patience = event["Patience"]

    return {
        "TaskName": input_items[0]["TaskName"],
        "NumOfClients": num_of_clients,
        "BatchSize": batch_size,
        "EpochCount": epoch_count,
        "Patience": patience,
        "VFLBucket": vfl_bucket,
    }
