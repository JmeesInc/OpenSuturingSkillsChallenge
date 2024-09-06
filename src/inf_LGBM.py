import pandas as pd
from lightgbm import LGBMRegressor
import sys
import os
import glob

input_dir = sys.argv[1]
file = sys.argv[2]
mode = sys.argv[3]
output_dir = sys.argv[4]

coords = sorted(glob.glob(input_dir + '/*.npy'))
columns = [x.split('/')[-1].split('.')[0] for x in coords]


df = pd.read_csv(file)
task1_target = df.loc[:, df.columns.str.startswith("gt_score")].mean(axis=1)
task2_target = df.loc[:, df.columns.str.startswith("gt_ostats")]

import pickle

if mode == "GRS":
    with open("/app/src/weight/model1.pkl", "rb") as f:
        model1 = pickle.load(f)
    pred = model1.predict(df)
    #classify, 0: 
    GRS = []
    for n in pred:
        if n < 15.5:
            n = 0
        elif n < 23.5:
            n = 1
        elif n < 31.5:
            n = 2
        else:
            n = 3
        GRS.append(n)

    submit = pd.DataFrame({"VIDEO": columns, "GRS": GRS})
    submit.to_csv(output_dir + "/Jmees_task1.csv", index=False)

elif mode == "OSATS":
    with open("/app/src/weight/model2.pkl", "rb") as f:
        model2 = pickle.load(f)
    pred = model2.predict(df)
    submit = pd.DataFrame({"VIDEO": columns, 
                            "OSATS_RESPECT": pred[:, 0],
                            "OSATS_MOTION": pred[:, 1],
                            "OSATS_INSTRUMENT": pred[:, 2],
                            "OSATS_SUTURE": pred[:, 3],
                            "OSATS_FLOW": pred[:, 4],
                            "OSATS_KNOWLEDGE": pred[:, 5],
                            "OSATS_PERFORMANCE": pred[:, 6],
                            "OSATSFINALQUALITY": pred[:, 7]})
    submit.to_csv(output_dir +  "/Jmees_task2.csv", index=False)

