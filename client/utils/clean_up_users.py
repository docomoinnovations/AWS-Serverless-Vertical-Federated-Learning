import boto3
from argparse import ArgumentParser


def get_iam_group_name(stack_name: str, region=None):
    client = boto3.client("cloudformation", region_name=region)
    stack = client.describe_stacks(StackName=stack_name)["Stacks"][0]

    outputs = stack["Outputs"]
    iam_group_name = ""
    for output in outputs:
        if output["OutputKey"] == "IAMGroupNameForClient":
            iam_group_name = output["OutputValue"]

    return iam_group_name


def list_iam_users_in_group(group_name: str):
    users = []
    client = boto3.client("iam")
    is_next = True
    marker = None

    while is_next:
        res = {
            "Users": [],
            "IsTruncated": False,
        }
        if marker:
            res = client.get_group(
                GroupName=group_name,
                Marker=marker,
            )
        else:
            res = client.get_group(
                GroupName=group_name,
            )
        for user in res["Users"]:
            users.append(user["UserName"])
        is_next = res["IsTruncated"]
        if is_next:
            marker = res["Marker"]

    return users


def delete_all_access_keys(user_name: str) -> None:
    access_key_ids = []
    client = boto3.client("iam")
    is_next = True
    marker = None

    while is_next:
        res = {
            "AccessKeyMetadata": [],
            "IsTruncated": False,
        }
        if marker:
            res = client.list_access_keys(
                UserName=user_name,
                Marker=marker,
            )
        else:
            res = client.list_access_keys(
                UserName=user_name,
            )

        for meta_data in res["AccessKeyMetadata"]:
            access_key_ids.append(meta_data["AccessKeyId"])

        is_next = res["IsTruncated"]
        if is_next:
            marker = res["Marker"]

    for access_key_id in access_key_ids:
        client.delete_access_key(
            UserName=user_name,
            AccessKeyId=access_key_id,
        )


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

    iam_group_name = get_iam_group_name(stack_name=stack_name, region=region)
    iam_users = list_iam_users_in_group(group_name=iam_group_name)

    if len(iam_users) == 0:
        exit(f'User not found in IAM group "{iam_group_name}".')

    confirmed = False

    while not confirmed:
        for iam_user in iam_users:
            print(iam_user)
        print(
            f'The above {len(iam_users)} users were found in "{iam_group_name}" group. Delete them permanetly? (y/n)',
            end=":",
        )
        answer = input()
        if answer.lower() == "y" or answer.lower() == "yes":
            confirmed = True
        elif answer.lower() == "n" or answer.lower() == "no":
            exit("Clean up was canceled.")
        else:
            exit("Canceled the clean up.")

    client = boto3.client("iam")

    for user in iam_users:
        delete_all_access_keys(user)
        client.remove_user_from_group(
            GroupName=iam_group_name,
            UserName=user,
        )
        client.delete_user(
            UserName=user,
        )
        print(f'User "{user}" deletion completed.')

    print(f'{len(iam_users)} users in "{iam_group_name}" group have been deleted.')
