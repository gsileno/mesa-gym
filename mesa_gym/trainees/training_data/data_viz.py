experiment_file = "zzt-random_100.pickle"

import pickle
with open(experiment_file, "rb") as f:
    data = pickle.load(f)

data_fields = data["fields"]
del(data["fields"])

# data processing for visualization

rows = []
for episode in data.keys():
    for step in data[episode].keys():
        for agent in data[episode][step]:
            row = {}
            row["agent"] = agent
            row["episode"] = episode
            row["step"] = step
            row["reward"] = data[episode][step][agent]["reward"]
            for key in data_fields:
                if key in data[episode][step][agent]:
                    row[key] = data[episode][step][agent][key]
                else:
                    row[key] = 0
            rows.append(row)

import pandas as pd
df = pd.DataFrame(rows)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

episodes = df.episode.unique()
agents = df.agent.unique()

fig, axs = plt.subplots(ncols=len(data_fields), nrows=len(agents), figsize=(12, 5))

for i, agent in enumerate(agents):
    x = []
    y = {}
    for key in data_fields:
        y[key] = []

    for episode in episodes:
        x.append(episode)
        Y = df[(df["agent"] == agent) & (df["episode"] == episode)][data_fields].sum()
        for key in data_fields:
            y[key].append(Y[key])

    # Plot outcomes
    for j, key in enumerate(data_fields):
        axs[i][j].set_title(f"agent {agent}")
        axs[i][j].set_xlabel("episode number")
        axs[i][j].set_ylabel(key)
        axs[i][j].plot(x, y[key])

plt.tight_layout()
plt.show()
