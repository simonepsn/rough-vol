This repository contains the material and code used for my master’s thesis "Roughly Speaking, on the Empirical Texture of Volatility".  
The study examines the empirical properties of volatility under the historical measure P, comparing classical econometric frameworks with rough volatility models based on fractional Brownian motion.

Objective: assess whether volatility is better represented as a rough, anti-persistent process rather than by mean-reverting or long-memory models.
Models: comparison between GARCH(1,1), HAR-RV, and Rough Fractional Stochastic Volatility (RFSV).
Estimation:
    GARCH(1,1): rolling-window maximum likelihood estimation of $(\omega, \alpha, \beta)$.
    HAR-RV: rolling OLS on log-realized volatility with daily, weekly, and monthly components.
    RFSV: moment-based estimation of the Hurst parameter $H$ and volatility scale $\nu$ via log–log regression on absolute increments.

Data: realized variance constructed from intraday prices (5-minute, hourly, and daily frequencies) for a panel of major U.S. stocks and the S\&P 500.
Evaluation: forecasting performance under $\mathbb{P}$ using RMSE and CRPS metrics, together with structural diagnostics based on autocorrelation and scaling behavior. 
Findings: rough models reproduce the empirical texture of volatility more accurately across frequencies, capturing irregularity and anti-persistence that traditional models fail to represent.


Have fun!
