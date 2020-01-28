import numpy as np
import pandas as pd

dict_x1_y = {
    "a": 0,
    "b": 1,
    "c": -1
}
x1 = np.random.choice(["a", "b", "c"], replace=True, size=1000)


df = pd.DataFrame({
    "x1": x1,
    "y": [dict_x1_y[i] + np.random.uniform(-0.7, 0.7) for i in x1]
})

df.head(800).to_csv("data/train.csv", index=False)
df.tail(200).to_csv("data/test.csv", index=False)