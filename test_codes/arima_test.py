from pandas import read_csv
from matplotlib import pyplot
from datetime import datetime
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

date_rng = pd.date_range(start='01/01/2018', end='12/31/2018', freq='H')
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = np.random.randint(0,100,size=(len(date_rng)))

df.describe()


print(df)
df.set_index("date", inplace=True)

model = ARIMA(df, order=(5,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())
'''
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())
series.plot()
pyplot.show()
'''