import pandas as pd

url = "https://raw.githubusercontent.com/sharmaroshan/Churn-Modelling-Dataset/master/Churn_Modelling.csv"
df = pd.read_csv(url)
df.to_csv("Churn_Modelling.csv", index=False)
print("Kích thước dữ liệu:", df.shape)