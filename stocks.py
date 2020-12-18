"""
Created on Sat Nov 28 15:39:55 2020

@author: smridhi
"""

from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as sct

# Step 1 - Import data
# Book-to-market ratios for the 25-size sorted portfolios
BM = pd.read_csv('ff25bm.csv', index_col=0, header=0) 
BM.index = pd.to_datetime(BM.index).to_period('M')

# log the BMs
BM.apply(np.log, inplace=True)

# Fama-French 3 factors
FF = pd.read_csv('FF3.csv', index_col=0, header=0) # returns in percentage
FF.index = pd.to_datetime(FF.index).to_period('M')

RF = FF['RF'] #Convert returns to rates because BMs are ratios
FF.drop('RF', axis=1, inplace=True)
RF1 = RF['1999-12':]

# Pull data 
time = TimeSeries(key='SBVUN3B758QVBOIK',output_format='pandas')

Amazon = time.get_monthly_adjusted(symbol = 'AMZN')
Microsoft = time.get_monthly_adjusted(symbol = 'MSFT')
Pfizer = time.get_monthly_adjusted(symbol = 'PFE')
Bank_of_America = time.get_monthly_adjusted(symbol = 'BAC')
Apple = time.get_monthly_adjusted(symbol = 'AAPL')

# Take out Closing price and reducing them from RF to make those excess return
Amazon1 = Amazon[0]['5. adjusted close']
Amazon2 = Amazon1.pct_change()*100
Amazon2.index = pd.to_datetime(Amazon2.index).to_period('M')
Amazon3 = pd.DataFrame(index = Amazon2.index,data=Amazon2 - RF1).dropna()

Microsoft1 = Microsoft[0]['5. adjusted close']
Microsoft2 = Microsoft1.pct_change()*100
Microsoft2.index = pd.to_datetime(Microsoft2.index).to_period('M')
Microsoft3 = pd.DataFrame(index = Microsoft2.index,data=Microsoft2 - RF1).dropna()

Pfizer1 = data=Pfizer[0]['5. adjusted close']
Pfizer2 = Pfizer1.pct_change()*100
Pfizer2.index = pd.to_datetime(Pfizer2.index).to_period('M')
Pfizer3 = pd.DataFrame(data=Pfizer2 - RF1).dropna()

Bank_of_America1 = Bank_of_America[0]['5. adjusted close']
Bank_of_America2 = Bank_of_America1.pct_change()*100
Bank_of_America2.index = pd.to_datetime(Bank_of_America2.index).to_period('M')
Bank_of_America3 = pd.DataFrame(data=Bank_of_America2 - RF1).dropna()

Apple1 = Apple[0]['5. adjusted close']
Apple2 = Pfizer1.pct_change()*100
Apple2.index = pd.to_datetime(Apple2.index).to_period('M')
Apple3 = pd.DataFrame(data=Apple2 - RF1).dropna()

print()
print('*'*30, "Adjusted Closing Price", '*'*30)
print()
amzplot = Amazon1.plot(title = 'AMAZON', color = 'red')
plt.show()
msftplot = Microsoft1.plot(title = 'Microsoft', color = 'green')
plt.show()
pfizerplot = Pfizer1.plot(title = 'Pfizer', color = 'purple')
plt.show()
boaplot = Bank_of_America1.plot(title = 'Bank of America', color = 'blue')
plt.show()
appleplot = Apple1.plot(title = 'Apple', color = 'black')
plt.show()

#Combining all data into 1 single dataframe
amzn_to_list = Amazon3[0].tolist()
msft_to_list = Microsoft3[0].tolist()
pfz_to_list = Pfizer3[0].tolist()
boa_to_list = Bank_of_America3[0].tolist()
aapl_to_list = Apple3[0].tolist()

test = { "Amazon":  amzn_to_list,
        "Microsoft": msft_to_list,
           "Pfizer": pfz_to_list,
           "Bank of America": boa_to_list,
           "Apple": aapl_to_list}

Data = pd.DataFrame(data = test, index = Microsoft3.index ,
                columns = ['Amazon','Microsoft','Pfizer', 'Bank of America', 'Apple'] )

# Flipping the timeline
Data1 = Data.iloc[::-1]

# training sample 
training_start = '1999-12'
training_end = '2019-12'

# test sample 
test_start = '2020-01'
test_end = '2020-08'

# training sample to parameter estimation
training_sample = Data1.loc[training_start:training_end,].index

# test sample
test_sample = Data1.loc[test_start:test_end,].index

# Step 2 - Run a Regression
varx = sm.tsa.VAR(endog=Data1.loc[training_sample[1:], :],
                  exog=BM.shift(1).loc[training_sample[1:], :]).fit(maxlags=0)
        
# Residuals
Sigma = varx.resid.cov()

# Step 3 - predictions (on the test sample)
direct_forecasts = pd.DataFrame(
    index=test_sample,
    columns=Data1.columns,
    data=sm.add_constant(BM.loc[test_sample-1,:]).values.dot(varx.params.values))

# Co-variance matrix
invcov_direct = pd.DataFrame(data=np.linalg.inv(Sigma),
                             index=Sigma.index,
                             columns=Sigma.columns)

risk_aversion_gamma = 10

# Step 4 - Optimal portfolio
def opt_port(mu, invcov, gamma):
    # risky weights
    risky_weights = (1/gamma)*invcov.dot(mu)
    # riskfree weight
    riskfree_weight = 1 - risky_weights.sum()
    weights = pd.Series(data=risky_weights.values, index=mu.index)
    weights.loc['riskfree'] = riskfree_weight
    return weights

# Direct portfolio weights
direct_weights =  direct_forecasts.apply(
    lambda x: opt_port(mu=x, invcov=invcov_direct, gamma=risk_aversion_gamma),
    axis=1
    )

# portfolio performance
OOSport = pd.DataFrame(index=test_sample, columns=['Direct'],
                       data=np.nan)
OOSport.loc[:, 'Direct'] = (direct_weights * Data1.join(RF).rename(
    {'RF':'riskfree'}, axis=1).loc[test_sample,:]).sum(axis=1)


def perf_res(srs, RF):
    import pandas as pd
    results = pd.Series(data=(
        srs.min(), # min in percent
        srs.mean()*12, # mean in annualzied percent
        srs.max(), # max in percent
        (srs-RF.loc[srs.index]).mean()/(srs-RF.loc[srs.index]).std()*np.sqrt(12)),
        index=['Min (%)', 'Mean (ann. %)', 'Max (%)', 'SR (ann.)']
        )
    return results

port_performance = (OOSport.apply(lambda x: perf_res(x, RF), axis=0))

# Evaluation
print()
print('*'*30, "Evaluation of Optimal Portfolio", '*'*30)
print()
print(port_performance)
print("Maximum weight",direct_weights .max().max())
print("Minimum weight",direct_weights.min().min())
print()
print(direct_weights.plot(title = "Weights of Optimal portfolio"))

# Step 5 - Comparision - Forecast v/s actual

print(direct_forecasts.plot(title = "Forecast returns"))

print( Data1['2020-01':'2020-08'].plot(title = 'Actual Returns'))



