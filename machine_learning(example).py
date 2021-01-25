import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
oecd_bli = pd.read_csv("oecd_bli_2019.csv",thousands=",")
gdp_per_capita = pd.read_csv("gdp_per_capita.csv",thousands=",",delimiter="\t",encoding="latin1",na_values="n/a")
country_stats = prepare_country_stats(oecd_bli,gdp_per_capita)
x = np.c_(country_stats["gdp_per_capita"])
y = np.c_(country_stats["Life Satisfaction"])
country_stats.plot(kind="scatter",x="gdp_per_capita",y="Life Satisfaction")
plt.show()
lin_reg_model = sklearn.linear_model.LinearRegression()
lin_reg_model.fit(x,y)
