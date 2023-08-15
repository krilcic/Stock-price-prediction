# Stock price prediction

## Summary

This is a program I made as a project for the subject of Computer modeling in the programming language Julia.
The task was to use the random walk model to predict the prices of a stock, but I also added the implementation of the [ARIMA model](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average).

Kicking off the project, I initiated the process by assessing the stationarity of the dataset through the application of the *Augmented Dickey-Fuller* test. This was followed by an analysis utilizing the *autocorrelation function (ACF)* and *partial autocorrelation function (PACF)* plots. These insights paved the way for identifying the optimal parameters driving the autoregressive integrated moving average (ARIMA) model.

![acf-pacf](https://github.com/krilcic/Stock-price-prediction/blob/master/Images/acf_pacf.png)

Building upon the preceding analyses, I concluded that an *ARIMA(4,1,4)* configuration was optimal, and thus proceeded to implement this model.

## Results

![ARIMAModel](https://github.com/krilcic/Stock-price-prediction/blob/master/Images/ARIMA(4%2C1%2C4)%20graph.png)

This is a comparison between the real price data and the prices given by my implementation of the ARIMA model which shows a good correlation. This tells me that the choice of parameters I used in my model was fairly good and gives me the green light to continue with the model.

#

![ARIMAModel](https://github.com/krilcic/Stock-price-prediction/blob/master/Images/ARIMA(4%2C1%2C4)%20prediction.png)

In the subsequent endeavor, my ARIMA model was set free to forecast future stock prices. However, the outcome paints a nuanced picture, indicating that the ARIMA model might not be sufficiently adept for highly volatile datasets such as the one used. The model's efficacy is often more pronounced in datasets characterized by modest volatility and discernible trends such as those found in business sales growth scenarios.

To enhance predictive performance, potential avenues include the integration of supplementary data sources, such as *median* and *box plot* analyses. These sources offer insights into significant zones of price reaction, as demonstrated below.

![ARIMAModel](https://github.com/krilcic/Stock-price-prediction/blob/master/Images/median.png)
![ARIMAModel](https://github.com/krilcic/Stock-price-prediction/blob/master/Images/boxplot.png)
