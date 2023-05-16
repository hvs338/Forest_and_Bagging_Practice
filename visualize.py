import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_training(model, x, y, title):

    data = pd.read_csv("iris_data.csv")
    colors = ['#747FE3', '#8EE35D', '#E37346']
    x = np.linspace(0, 30, 100)
    y = np.linspace(0, 16, 100)
    xv, yv = np.meshgrid(x, y)

    matr = np.asarray([xv.flatten(), yv.flatten()]).T
    predictions = model.predict(X=matr)
    over_lay = pd.DataFrame(matr, columns=["Sepal Area", "PetalArea"])
    over_lay["Species"] = predictions

    fig, ax = plt.subplots()
    sns.scatterplot(x="Sepal Area", y="PetalArea", hue="Species", data=data, palette=colors, ax=ax).set(title=title)
    ax2 = ax.twinx()
    sns.scatterplot(x="Sepal Area", y="PetalArea", hue="Species", data=over_lay, palette=colors, ax=ax2, alpha=0.03,
                    legend=False)

    plt.show()
    fig.savefig(title + ".pdf")

    return None