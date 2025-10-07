This repository contains the material and code used for the master’s thesis "Roughly Speaking, on the Empirical Texture of Volatility.  
The study examines the empirical properties of volatility under the historical measure P, comparing classical econometric frameworks with rough volatility models based on fractional Brownian motion.

\begin{itemize}
    \item \textbf{Objective:} assess whether volatility is better represented as a rough, anti-persistent process rather than by mean-reverting or long-memory models.
    \item \textbf{Models:} comparison between GARCH(1,1), HAR-RV, and Rough Fractional Stochastic Volatility (RFSV).
    \item \textbf{Estimation:} 
    \begin{itemize}
        \item GARCH(1,1): rolling-window maximum likelihood estimation of $(\omega, \alpha, \beta)$.
        \item HAR-RV: rolling OLS on log-realized volatility with daily, weekly, and monthly components.
        \item RFSV: moment-based estimation of the Hurst parameter $H$ and volatility scale $\nu$ via log–log regression on absolute increments.
    \end{itemize}
    \item \textbf{Data:} realized variance constructed from intraday prices (5-minute, hourly, and daily frequencies) for a panel of major U.S. stocks and the S\&P 500.
    \item \textbf{Evaluation:} forecasting performance under $\mathbb{P}$ using RMSE and CRPS metrics, together with structural diagnostics based on autocorrelation and scaling behavior.
    \item \textbf{Findings:} rough models reproduce the empirical texture of volatility more accurately across frequencies, capturing irregularity and anti-persistence that traditional models fail to represent.
\end{itemize}

Have fun!
