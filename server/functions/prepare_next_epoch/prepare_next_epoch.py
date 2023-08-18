def lambda_handler(event, context):
    print(event)

    response = []
    for item in event:
        item["BatchIndex"] = 0
        item["VaBatchIndex"] = 0
        item["EpochIndex"] = int(item["EpochIndex"]) + 1
        item["IsNextBatch"] = int(item["BatchIndex"]) + 1 < int(item["BatchCount"])
        item["IsNextVaBatch"] = int(item["VaBatchIndex"]) + 1 < int(
            item["VaBatchCount"]
        )
        item["IsNextEpoch"] = int(item["EpochIndex"]) + 1 < int(item["EpochCount"])

        response.append(item)

    return response
