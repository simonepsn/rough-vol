This repository contains the codebase for an empirical comparison of three volatility models: GARCH(1,1), HAR-RV, and a lognormal Rough Fractional Stochastic Volatility (RFSV) model—estimated and evaluated under the historical measure P. 

Using a daily log-volatility series computed from high-frequency price data, each model is calibrated to match the observed dynamics of realized variance. GARCH is estimated via likelihood, HAR through regression on lagged realized volatility, and RFSV via simulation of log-volatility paths driven by fractional Brownian motion with Hurst parameter H ≅ 0.1, fitted using a moment-based strategy. Forecast accuracy and structural fit are assessed by comparing simulated and actual series across multiple statistical metrics. 

The RFSV simulation relies on a Volterra integral discretization consistent with the theoretical formulation of rough volatility, following the approach of Bayer et al. (2016), Bennedsen et al. (2017) and Gatheral et al. (2018).
