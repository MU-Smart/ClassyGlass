import glob
import pandas as pd
import numpy as np
import os



# --- Config ---

WINDOW_SIZE = 500
STEP_SIZE = WINDOW_SIZE // 2  # 50% overlap

DATASET_PATH = "../Datasets/Dataset_1A"
OUTPUT_DIR = f"../DatasetsFeatureExtracted/Dataset_1A/{WINDOW_SIZE}_{STEP_SIZE}"

expDic = {
    1: "Sitting and Reading a book",
    2: "Sitting and Writing on a notebook",
    3: "Using computer (Typing)",
    4: "Using computer (Browsing)",
    5: "While sitting (Moving head, body)",
    6: "While sitting (Moving chair)",
    7: "Sitting (Stand up from sitting)",
    8: "Standing",
    9: "Walking",
    10: "Running",
    11: "Taking stairs",
}


def get_activity_id(exp_id):
    if exp_id % 11 == 0:
        return 11
    return exp_id % 11


def get_activity_name(activity_id):
    return expDic[activity_id]


def extract_features(window_acc, window_gyr, exp_id, activity_id, activity_name):
    acc_sumxyz = window_acc.sum()
    acc_abssum = window_acc.abs().sum()

    gyr_sumxyz = window_gyr.sum()
    gyr_abssum = window_gyr.abs().sum()

    return {
        "exp_id": exp_id,
        "activity_id": activity_id,
        "activity_name": activity_name,
        # Accelerometer features
        "acc_x_mean": window_acc["x"].mean(),
        "acc_x_var": window_acc["x"].var(),
        "acc_y_mean": window_acc["y"].mean(),
        "acc_y_var": window_acc["y"].var(),
        "acc_z_mean": window_acc["z"].mean(),
        "acc_z_var": window_acc["z"].var(),
        "acc_sum_mean": acc_sumxyz.mean(),
        "acc_abssum_mean": acc_abssum.mean(),
        "acc_sum_var": acc_sumxyz.var(),
        "acc_abssum_var": acc_abssum.var(),
        "acc_maxabssum": acc_abssum.max(),
        # Gyroscope features
        "gyr_x_mean": window_gyr["x"].mean(),
        "gyr_x_var": window_gyr["x"].var(),
        "gyr_y_mean": window_gyr["y"].mean(),
        "gyr_y_var": window_gyr["y"].var(),
        "gyr_z_mean": window_gyr["z"].mean(),
        "gyr_z_var": window_gyr["z"].var(),
        "gyr_sum_mean": gyr_sumxyz.mean(),
        "gyr_abssum_mean": gyr_abssum.mean(),
        "gyr_sum_var": gyr_sumxyz.var(),
        "gyr_abssum_var": gyr_abssum.var(),
        "gyr_maxabssum": gyr_abssum.max(),
    }


# --- Load file index ---
file_list = glob.glob(f"{DATASET_PATH}/*/*.csv")
df = pd.DataFrame(file_list, columns=["path"])
df["file"] = df["path"].apply(lambda x: x.split("/")[-1])
df["user"] = df["path"].apply(lambda x: x.split("/")[-2])
df["expID"] = df["file"].apply(lambda x: int(x.split("_")[0]))
df["sensor"] = df["file"].apply(lambda x: x.split("_")[4])

df.sort_values(by="expID", inplace=True)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Process per user ---
for user in sorted(df["user"].unique()):
    user_df = df[df["user"] == user]
    result_list = []

    for exp_id in sorted(user_df["expID"].unique()):
        temp_df = user_df[user_df["expID"] == exp_id]

        # Skip if required sensors are missing
        if not {"Accelerometer", "Gyroscope"}.issubset(set(temp_df["sensor"])):
            continue

        # Skip TUG test (different pattern, only 1 sample)
        if exp_id % 6 == 0:
            continue

        activity_id = get_activity_id(exp_id)
        activity_name = get_activity_name(activity_id)

        acc_path = temp_df[temp_df["sensor"] == "Accelerometer"].iloc[0]["path"]
        gyr_path = temp_df[temp_df["sensor"] == "Gyroscope"].iloc[0]["path"]

        acc_data = pd.read_csv(acc_path)
        acc_data.rename(
            columns={"x-axis (g)": "x", "y-axis (g)": "y", "z-axis (g)": "z"},
            inplace=True,
        )
        acc_data = acc_data[["x", "y", "z"]]

        gyr_data = pd.read_csv(gyr_path)
        gyr_data.rename(
            columns={
                "x-axis (deg/s)": "x",
                "y-axis (deg/s)": "y",
                "z-axis (deg/s)": "z",
            },
            inplace=True,
        )
        gyr_data = gyr_data[["x", "y", "z"]]

        for i in range(0, len(acc_data), STEP_SIZE):
            window_acc = acc_data[i : i + WINDOW_SIZE]
            if len(window_acc) < WINDOW_SIZE:
                break
            window_gyr = gyr_data[i : i + WINDOW_SIZE]

            result_list.append(
                extract_features(window_acc, window_gyr, exp_id, activity_id, activity_name)
            )

    if result_list:
        user_result_df = pd.DataFrame(result_list)
        out_path = os.path.join(OUTPUT_DIR, f"{user}_features_W{WINDOW_SIZE}_O50.csv")
        user_result_df.to_csv(out_path, index=False)
        print(f"[{user}] {len(user_result_df)} windows → {out_path}")
    else:
        print(f"[{user}] No valid data found, skipping.")

print("\nDone! Per-user CSVs ready for LOSO cross-validation.")
