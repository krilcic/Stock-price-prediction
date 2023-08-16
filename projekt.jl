using CSV
using DataFrames
using Plots
using LinearAlgebra
using Statistics
using Distributions
using Optim
using StatsPlots
using TimeSeries
using StateSpaceModels
using HypothesisTests
using StatsBase
using StatsModels

# saving data to variable data
data = CSV.read("TSLA.csv", DataFrame, types = [Float64, Float64, Float64, Float64, Float64, Float64, Int64])

# saving daily closing prices to variable close_prices
close_prices = data.Close

# -------------------- Random walk model --------------------

# implemetation of random walk model
function random_walk_model(price_series)
	len = length(price_series)
	# vector for saving predicted prices
	pred_prices = zeros(len)
	# setting the first price to be the same as the price from input data
	pred_prices[1] = price_series[1]

	# predicting future prices
	for i in 2:len
		pred_prices[i] = pred_prices[i-1] + 10 * randn()
	end

	return pred_prices
end

# calling the function for random walk model
random_price = random_walk_model(close_prices)

# visualization of real and predicted values
plot(close_prices, label = "Real graph", xlabel = "Day", ylabel = "Price")
plot!(random_price, label = "Random Walk graph")



# -------------------- ARIMA model --------------------

# doing the Augmented Dickey-Fuller test for stationarity of the data vector
ADFTest(close_prices, Symbol("constant"), 4)

# since p-value > 0.05, we can't reject the null hypothesis and we conclude that the data vector is not stationary
diff_price = diff(close_prices)

# repeating the Augmented Dickey-Fuller test for stationarity of the data vector
ADFTest(diff_price, Symbol("constant"), 4)

# making acf and pacf plots to determine the parameters of the model
total_lags = 20
scatter1 = scatter(collect(1:total_lags), autocor(diff_price, collect(1:total_lags)), title = "ACF", ylim = [-0.3, 0.5])
scatter2 = scatter(collect(1:total_lags), pacf(diff_price, collect(1:total_lags)), title = "PACF", ylim = [-0.3, 0.5])
plot(scatter1, scatter2, layout = (2, 1))

# implementation of ARIMA(4,1,4) model
function arima414(θ, x)
	# length of vector containing data
	len = length(x)
	# vector for saving predicted prices
	y = zeros(len)
	# saving starting values
	y[1] = x[1]
	y[2] = x[2]
	y[3] = x[3]
	y[4] = x[4]
	# saving parameters
	ϕ1, ϕ2, ϕ3, ϕ4, μ, θ1, θ2, θ3, θ4 = θ

	for t in 5:len
		ytm1 = y[t-1]
		ytm2 = y[t-2]
		ytm3 = y[t-3]
		ytm4 = y[t-4]

		y[t] = μ + ytm1 + # general starting value ("I" part of ARIMA)
			   ϕ1 * (ytm1 - ytm2) + ϕ2 * (ytm2 - ytm3) + ϕ3 * (ytm3 - ytm4) + ϕ4 * (ytm4 - ytm1) + # AR(4) part - auto-regression
			   θ1 * (x[t-1] - ytm2) + θ2 * (x[t-2] - ytm3) + θ3 * (x[t-3] - ytm4) + θ4 * (x[t-4] - ytm1) # MA(4) part - moving average    
	end

	return y
end

# function for negative log-likelihood
function negloglik(θ, x)
	# calling the function for ARIMA(4,1,4) model
	y = arima414(θ, x)
	# calculating residuals
	residuals = x[1:end] - y[1:end]
	# length of vector containing data
	len = length(x)

	# calculating negative log-likelihood
	return 0.5 * len * log(2 * pi) + 0.5 * sum(residuals .^ 2)
end

# optimization for finding parameters that minimize negative log-likelihood
optimization = optimize(θ -> negloglik(θ, close_prices), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
ϕ1, ϕ2, ϕ3, ϕ4, μ, θ1, θ2, θ3, θ4 = optimization.minimizer

# saving parameters to vector and calling the function for ARIMA(4,1,4) model
θ = ϕ1, ϕ2, ϕ3, ϕ4, μ, θ1, θ2, θ3, θ4
result = arima414(θ, close_prices)

# vizualization of real and predicted values
plot(result, label = "ARIMA(4,1,4) graph", xlabel = "Day", ylabel = "Price")
plot!(close_prices, label = "Real graph")

# function for predicting future values
function arima414_predictions(θ, x, num_predictions)
	# saving first 150 values of the vector containing data
	y = x[1:150]
	# length of vector containing data
	len = length(y)
	# saving parameters
	ϕ1, ϕ2, ϕ3, ϕ4, μ, θ1, θ2, θ3, θ4 = θ

	for i in 1:num_predictions
		t = len + i
		ytm1 = y[t-1]
		ytm2 = y[t-2]
		ytm3 = y[t-3]
		ytm4 = y[t-4]
		# formula for predicting future values(similar to the formula for ARIMA(4,1,4) model, but uses predicted value instead of real values)
		yi = μ + ytm1 +
			 ϕ1 * (ytm1 - ytm2) + ϕ2 * (ytm2 - ytm3) + ϕ3 * (ytm3 - ytm4) + ϕ4 * (ytm4 - ytm1) +
			 θ1 * (y[t-1] - ytm2) + θ2 * (y[t-2] - ytm3) + θ3 * (y[t-3] - ytm4) + θ4 * (y[t-4] - ytm1)

		append!(y, yi)
	end

	return y
end

# calling the function for predicting future values and visualization
y414 = arima414_predictions(θ, close_prices, 100)

plot!(y414, label = "ARIMA(4,1,4) prediction")

# calculating descriptive statistics
med = median(close_prices)
std_dev = std(close_prices)
quant = quantile(close_prices, [0.25, 0.5, 0.75])

# creating a graph of stock prices
plot(close_prices, label = "Stock prices")

# adding line for median
hline!([med], line = :dash, color = :lightgreen)

# adding label for median
annotate!(med, med, text("Median", :top, :right))

# Box plot
boxplot(close_prices, ylabel = "Value", title = "Box plot")
