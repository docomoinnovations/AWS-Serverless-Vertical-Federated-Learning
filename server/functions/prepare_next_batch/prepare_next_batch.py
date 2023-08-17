def lambda_handler(event, context):
    print(event)
    response = []
    for item in event["InputItems"]:
        item["BatchIndex"] = int(item["BatchIndex"]) + 1
        item["IsNextBatch"] = int(item["BatchIndex"]) + 1 < int(item["BatchCount"])

        response.append(item)

    return response
