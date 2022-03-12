import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pred_test.csv")
plt.scatter(df['prediction'], df['test'])

plt.show()
