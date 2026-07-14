import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

# sns.scatterplot(
#     data=tips,
#     x="total_bill",
#     y="tip",
#     hue="day"
# )

# plt.xlabel("Tips")
# plt.ylabel("Total Bill")
# plt.grid()
# plt.title("Restraunt Tips")
# plt.show()

# sns.countplot(
#     data=tips,
#     x="day",
#     hue="smoker"
# )

# sns.histplot(
#     data=tips,
#     x="total_bill",
#     bins=40,
#     kde=True
# )

# sns.boxplot(
#     data=tips,
#     x="day",
#     y="total_bill",
#     hue="smoker"
# )

correlation = tips.corr(numeric_only=True)

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

# print(tips.head())
plt.show()