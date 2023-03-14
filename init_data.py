from pathlib import Path
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

TRAINING_DATA_SET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
TEST_DATA_SET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)

seed = 42


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_data():
    tr = pd.read_csv(TRAINING_DATA_SET_URL, header=None)
    te = pd.read_csv(TEST_DATA_SET_URL, header=None, skiprows=1)

    h = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "over_50k",
    ]

    tr.columns = h
    te.columns = h
    te.over_50k = te.over_50k.str.split(".", expand=True)[0]

    num_cols = [
        "age",
        "fnlwgt",
        "education_num",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
    ]
    cat_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    lab_col = "over_50k"

    return tr, te, num_cols, cat_cols, lab_col


def create_encoders(dat, cat_cols, num_cols):
    encoders = dict()

    # OneHotencoding for categorical values
    for c in cat_cols:
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit(dat[c].astype(str).values.reshape(-1, 1))
        encoders[c] = enc

    # MiniMaxScaler for numerical values
    for c in num_cols:
        scaler = MinMaxScaler()
        scaler.fit(dat[c].values.reshape(-1, 1))
        encoders[c] = scaler

    return encoders


def encode(dat, cat_cols, num_cols, encoders):
    for c in cat_cols:
        out = encoders[c].transform(dat[c].astype(str).values.reshape(-1, 1))
        if not type(out) == np.ndarray:
            out = out.todense()
        keys = [f"{c}_{i}" for i in range(out.shape[1])]
        dat[keys] = out
        dat = dat.drop(c, axis=1)

    for c in num_cols:
        out = encoders[c].transform(dat[c].values.reshape(-1, 1)).flatten()
        dat[c] = out

    return dat


def prepare_data():
    tr, te, num_cols, cat_cols, lab_col = load_data()
    tr, va = train_test_split(tr, test_size=0.1, random_state=seed)

    encoders = create_encoders(tr, cat_cols, num_cols)
    tr = encode(tr, cat_cols, num_cols, encoders)
    va = encode(va, cat_cols, num_cols, encoders)
    te = encode(te, cat_cols, num_cols, encoders)

    tr["user_id"] = range(0, len(tr))
    tr = tr.set_index("user_id")
    va["user_id"] = range(len(tr), len(tr) + len(va))
    va = va.set_index("user_id")
    te["user_id"] = range(len(tr) + len(va), len(tr) + len(va) + len(te))
    te = te.set_index("user_id")

    tr[lab_col] = tr[lab_col].replace(
        {" <=50K": 0, " >50K": 1}
    )  # Be careful for white space
    va[lab_col] = va[lab_col].replace(
        {" <=50K": 0, " >50K": 1}
    )  # Be careful for white space
    te[lab_col] = te[lab_col].replace(
        {" <=50K": 0, " >50K": 1}
    )  # Be careful for white space

    tr_x = tr.drop(lab_col, axis=1)
    tr_y = tr[lab_col]
    va_x = va.drop(lab_col, axis=1)
    va_y = va[lab_col]
    te_x = te.drop(lab_col, axis=1)
    te_y = te[lab_col]

    tr_y = torch.FloatTensor(tr_y.values).reshape(-1, 1)
    va_y = torch.FloatTensor(va_y.values).reshape(-1, 1)
    te_y = torch.FloatTensor(te_y.values).reshape(-1, 1)

    return tr_x, tr_y, va_x, va_y, te_x, te_y


def get_client_cols(client):
    cols = []

    if client == 1:
        cols = [
            "age",
            "sex_0",
            "sex_1",
            "marital_status_0",
            "marital_status_1",
            "marital_status_2",
            "marital_status_3",
            "marital_status_4",
            "marital_status_5",
            "marital_status_6",
            "relationship_0",
            "relationship_1",
            "relationship_2",
            "relationship_3",
            "relationship_4",
            "relationship_5",
            "fnlwgt",
        ]

    elif client == 2:
        cols = [
            "race_0",
            "race_1",
            "race_2",
            "race_3",
            "race_4",
            "native_country_0",
            "native_country_1",
            "native_country_2",
            "native_country_3",
            "native_country_4",
            "native_country_5",
            "native_country_6",
            "native_country_7",
            "native_country_8",
            "native_country_9",
            "native_country_10",
            "native_country_11",
            "native_country_12",
            "native_country_13",
            "native_country_14",
            "native_country_15",
            "native_country_16",
            "native_country_17",
            "native_country_18",
            "native_country_19",
            "native_country_20",
            "native_country_21",
            "native_country_22",
            "native_country_23",
            "native_country_24",
            "native_country_25",
            "native_country_26",
            "native_country_27",
            "native_country_28",
            "native_country_29",
            "native_country_30",
            "native_country_31",
            "native_country_32",
            "native_country_33",
            "native_country_34",
            "native_country_35",
            "native_country_36",
            "native_country_37",
            "native_country_38",
            "native_country_39",
            "native_country_40",
            "native_country_41",
        ]

    elif client == 3:
        cols = [
            "workclass_0",
            "workclass_1",
            "workclass_2",
            "workclass_3",
            "workclass_4",
            "workclass_5",
            "workclass_6",
            "workclass_7",
            "workclass_8",
            "occupation_0",
            "occupation_1",
            "occupation_2",
            "occupation_3",
            "occupation_4",
            "occupation_5",
            "occupation_6",
            "occupation_7",
            "occupation_8",
            "occupation_9",
            "occupation_10",
            "occupation_11",
            "occupation_12",
            "occupation_13",
            "occupation_14",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
        ]

    elif client == 4:
        cols = [
            "education_0",
            "education_1",
            "education_2",
            "education_3",
            "education_4",
            "education_5",
            "education_6",
            "education_7",
            "education_8",
            "education_9",
            "education_10",
            "education_11",
            "education_12",
            "education_13",
            "education_14",
            "education_15",
            "education_num",
        ]

    return cols


def allocate_client_data(client, x):
    cols = get_client_cols(client)
    xt = torch.FloatTensor(x[cols].values)
    return xt


if __name__ == "__main__":
    tr_x, tr_y, va_x, va_y, te_x, te_y = prepare_data()
    tr_uid = torch.LongTensor(tr_x.index)
    va_uid = torch.LongTensor(va_x.index)
    te_uid = torch.LongTensor(te_x.index)

    np.save(
        "server/functions/server_training/tr_y.npy", tr_y.numpy(), allow_pickle=False
    )
    np.save(
        "server/functions/server_training/va_y.npy", va_y.numpy(), allow_pickle=False
    )
    np.save("test/te_y.npy", te_y.numpy(), allow_pickle=False)
    np.save(
        "server/functions/init_server/tr_uid.npy", tr_uid.numpy(), allow_pickle=False
    )
    np.save(
        "server/functions/server_training/tr_uid.npy",
        tr_uid.numpy(),
        allow_pickle=False,
    )
    np.save(
        "server/functions/init_server/va_uid.npy", va_uid.numpy(), allow_pickle=False
    )
    np.save(
        "server/functions/server_training/va_uid.npy",
        va_uid.numpy(),
        allow_pickle=False,
    )
    np.save("test/te_uid.npy", te_uid.numpy(), allow_pickle=False)

    for client in range(1, 5):
        client_cols = get_client_cols(client)
        client_tr_x = allocate_client_data(client, tr_x)
        client_va_x = allocate_client_data(client, va_x)
        client_te_x = allocate_client_data(client, te_x)

        Path(f"client/dataset/client{client}").mkdir(parents=True, exist_ok=True)

        np.save(
            f"client/dataset/client{client}/tr_uid.npy",
            tr_uid.numpy(),
            allow_pickle=False,
        )
        np.save(
            f"client/dataset/client{client}/va_uid.npy",
            va_uid.numpy(),
            allow_pickle=False,
        )
        np.save(
            f"client/dataset/client{client}/tr_x.npy",
            client_tr_x.numpy(),
            allow_pickle=False,
        )
        np.save(
            f"client/dataset/client{client}/va_x.npy",
            client_va_x.numpy(),
            allow_pickle=False,
        )
        np.save(
            f"client/dataset/client{client}/cols.npy",
            np.array(client_cols),
            allow_pickle=False,
        )
        np.save(f"test/cols_{client}.npy", np.array(client_cols), allow_pickle=False)
        np.save(f"test/te_x_{client}.npy", client_te_x.numpy(), allow_pickle=False)
