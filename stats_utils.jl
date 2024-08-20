### General status utility functions

# Bootstrap a correlation
function bootstrap_correlation(x, y, n_bootstrap=1000)
	n = length(x)
	corrs = Float64[]  # To store the bootstrap correlations

	for i in 1:n_bootstrap
		# Resample the data with replacement
		idxs = sample(Xoshiro(i), 1:n, n, replace=true)
		x_resample = x[idxs]
		y_resample = y[idxs]
		
		# Compute the correlation for the resampled data
		push!(corrs, cor(x_resample, y_resample))
	end

	return corrs
end