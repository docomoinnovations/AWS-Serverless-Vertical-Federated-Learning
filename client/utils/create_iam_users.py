import json
import random
import string
import boto3
from argparse import ArgumentParser

client_configs = [
    "../config/1.json",
    "../config/2.json",
    "../config/3.json",
    "../config/4.json",
]


def get_stack_attributes(stack_name: str, region=None):
    client = boto3.client("cloudformation", region_name=region)
    stack = client.describe_stacks(StackName=stack_name)["Stacks"][0]

    parameters = stack["Parameters"]
    num_of_clients = 0
    for param in parameters:
        if param["ParameterKey"] == "NumOfClients":
            num_of_clients = int(param["ParameterValue"])

    outputs = stack["Outputs"]
    iam_group_name = ""
    for output in outputs:
        if output["OutputKey"] == "IAMGroupNameForClient":
            iam_group_name = output["OutputValue"]

    return {
        "NumOfClients": num_of_clients,
        "IAMGroupName": iam_group_name,
    }


def get_random_str(n: int):
    return "".join(
        [random.choice(string.ascii_letters + string.digits) for i in range(n)]
    )


def user_exists(user_name: str):
    client = boto3.client("iam")
    try:
        client.get_user(
            UserName=user_name,
        )
        return True
    except client.exceptions.NoSuchEntityException:
        return False


if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument(
        "-s",
        "--stack-name",
        type=str,
        required=True,
        help="Specify CloudFormation stack name of VFL.",
    )
    argparser.add_argument(
        "-r",
        "--region",
        type=str,
        default=None,
        help="Specify region where CloudFormation stack is deployed.",
    )

    args = argparser.parse_args()
    stack_name = args.stack_name
    region = args.region

    attributes = get_stack_attributes(stack_name, region)

    num_of_clients = attributes["NumOfClients"]
    iam_group_name = attributes["IAMGroupName"]

    iam_client = boto3.client("iam")
    iam_users = []
    for i in range(num_of_clients):
        client_config = client_configs[i]

        suffix = get_random_str(5)
        iam_user_name = f"vfl-client-{suffix}"
        while user_exists(iam_user_name):
            suffix = get_random_str(5)
            iam_user_name = f"vfl-client-{suffix}"

        with open(client_config, "r") as f:
            config = json.load(f)
            client_id = config["member_id"]
            iam_client.create_user(
                UserName=iam_user_name,
                Tags=[
                    {
                        "Key": "vfl-client-id",
                        "Value": client_id,
                    }
                ],
            )

            accesskey = iam_client.create_access_key(
                UserName=iam_user_name,
            )

            access_key_id = accesskey["AccessKey"]["AccessKeyId"]
            secret_access_key = accesskey["AccessKey"]["SecretAccessKey"]

            iam_users.append(
                {
                    "ClientId": client_id,
                    "UserName": iam_user_name,
                    "AccessKeyId": access_key_id,
                    "SecretAccessKey": secret_access_key,
                }
            )

        iam_client.add_user_to_group(
            GroupName=iam_group_name,
            UserName=iam_user_name,
        )

    print("Run the following commands in each client terminal to set IAM credentials.")
    for iam_user in iam_users:
        title = "# Client ID: {} (IAM user: {}) #".format(
            iam_user["ClientId"], iam_user["UserName"]
        )
        print("")
        print("#" * len(title))
        print(title)
        print("#" * len(title))
        print('export AWS_ACCESS_KEY_ID="{}"'.format(iam_user["AccessKeyId"]))
        print('export AWS_SECRET_ACCESS_KEY="{}"'.format(iam_user["SecretAccessKey"]))
