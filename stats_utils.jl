### General stats utility functions

"""
Compute the Fisher Information Matrix for a Turing model and dataset at given parameter values.

# Arguments
- `model::DynamicPPL.Model`: The Turing model for which to compute the Fisher Information Matrix. Data should be provided in the model.
- `params::NamedTuple`: A named tuple containing parameter names and their current values.
- `summary_method::Function`: An optional function to summarize the Fisher Information Matrix. Defaults to `tr` (trace of the matrix). Possible alternate value: `det` (matrix determinant).

# Returns
- Summary of the Fisher Information Matrix. By default, this is the trace of the matrix.

# Details
The Fisher Information Matrix is computed as the negative Hessian of the log-likelihood function with respect to the parameters. The log-likelihood function is evaluated using the Turing model and the parameter values provided.
"""
function FI(
	model::DynamicPPL.Model,
	params::NamedTuple;
	summary_method::Function = tr
)

	# Exctract param names and value from NamedTuple
	param_names = keys(params)
	param_values = collect(values(params))

	# Define loglikelihood function for ForwardDiff, converting vector of parameter value needed by ForwardDiff.hessian to NamedTuple needed by Turing model
	ll(x) = loglikelihood(model, (;zip(param_names, x)...))

	# Compute FI as negative hessian
	FI = -ForwardDiff.hessian(ll, param_values)

	# Return trace
	return summary_method(FI)
end

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