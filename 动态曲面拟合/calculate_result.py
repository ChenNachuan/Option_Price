import os
import numpy as np
import pandas as pd
import json

config_path = "../config.json"
submission_file_path = "submission.csv"
OPTION_MONTH = "IO2507"

date = ""
data_path = ""

with open(config_path) as f:
    config = f.read()
    config_json = json.loads(config)
    date = config_json["date"]
    data_path = config_json["data_path"]

option_md_path = f"{data_path}/{OPTION_MONTH}_{date}.csv"
option_md_static_path = f"{data_path}/{OPTION_MONTH}_{date}_static.csv"

submit_df = pd.read_csv(submission_file_path)
option_md_df = pd.read_csv(option_md_path)
option_md_static_df = pd.read_csv(option_md_static_path)

na_count = submit_df['target'].isna().sum()

if na_count > 0 :
    print("存在nan,请检查代码")
    raise("输出结果存在nan")

option_md_df["mid_p"] = np.where(
    option_md_df["ask_p1"].notna() & option_md_df["bid_p1"].notna(),
    (option_md_df["ask_p1"] + option_md_df["bid_p1"]) / 2,
    np.where(
        option_md_df["ask_p1"].notna(),
        option_md_df["ask_p1"],
        option_md_df["bid_p1"]
    )
)

mae = (submit_df["target"] - option_md_df["mid_p"].shift(-len(option_md_static_df)))
mae = mae.dropna()
res = mae.abs().mean()

timing_file = "iter_test_timing.json"
running_time = 1000000000.0
if os.path.exists(timing_file):
    try:
        with open(timing_file, "r") as f:
            timing_data = json.load(f)
        running_time = timing_data['iter_test_running_time']
    except Exception as e:
        pass

print(f"{res:.6f},{running_time:.4f}")