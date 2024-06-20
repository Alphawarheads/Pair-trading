Analysis: economic intuition, quantitative analysis；

Data: universe, date range, data sets, databases；

Strat details: formulas for signal generation, portfolio construction, trx cost estimate

1. **Analysis:**

**Economic intuition:** The economic intuition behind using Bollinger Bands in a pairs trading strategy is based on exploiting temporary market inefficiencies and mean reversion, where price deviations from the historical average provide opportunities for profit as prices are expected to revert to their mean, leveraging both statistical significance and market psychology.

**Quantitative analysis**- we would conduct calculations on IR, Sharp ratio, Maximum Drawdown

1. **Data:**

**Universe:** Shanghai and Shenzhen 300 Index (CSI300) Data

**Date range:** From May 6, 2013 to present (training-validation-testing 80：10: 10 given the large amount of data)

**Data sets:** Closing, volume, percentage change, (Time Scale: daily)

**Data base:** From Eastmoney.com(东方财富网)

1. **Strat details**

**Signal Generation Formula:**

**Buy Signal: Xt​<μ(t)−kσ(t):** 

**Sell Signal: Xt​>μ(t)+kσ(t):**

**No Trade Condition: μ(t)−kσ(t)≤Xt​≤μ(t)+kσ(t)**: 

Xt​- represents the value of the spread at time t. 

μ(t)- the central line in the Bollinger bands.

k - A scaling factor used to set the width of the Bollinger bands.

σ(t) **-** is the standard deviation of the spread, indicating the degree of price volatility.

Spread: Let SAt and SBt represent the prices of stocks A and B at time t, respectively. 

Xt​ =ln⁡(SAtSA0)- ln⁡(SBtSB0)

**Portfolio Construction Formula**

The portfolio consists of the top 10 pairs selected based on their high mean-reversion speed and significant number of overnight jumps. These pairs are actively traded over a 5-day period, where trades are made based on the entry signals defined by the Bollinger bands strategy .

**Transaction Cost Estimate Formula**

Transaction costs are estimated at 5 basis points per share per half-turn. This estimate is factored into the performance calculations to provide a realistic view of the strategy’s profitability after accounting for trading costs .

Cost=Number of shares×Cost per share per half-turn

Bid-Ask Spread:0.05%

Impact Cost: 0.2%

Other Fees: 0.03%

Total Transaction Cost= 0.05% + 0.2% + 0.03% = 0.28% 

Reference: Stübinger, Johannes, and Endres, Sylvia. (2018). Pairs trading with a mean-reverting jump-diffusion model on high-frequency data.

