import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Dataset/datafile (1).csv")

# Top 10 yields
top_yield = df.sort_values(
    by="Yield (Quintal/ Hectare)",
    ascending=False
)

plt.figure(figsize=(10,5))
plt.bar(top_yield["State"][:10],
        top_yield["Yield (Quintal/ Hectare)"][:10])

plt.xticks(rotation=45)
plt.title("Top States by Yield")
plt.xlabel("State")
plt.ylabel("Yield")
plt.tight_layout()

plt.show()