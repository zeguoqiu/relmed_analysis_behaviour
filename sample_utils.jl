# Functions for models and working with draws from models

# Transform unconstrainted a to learning rate α
a2α(a) = logistic(π/sqrt(3) * a)

α2a(α) = logit(α) / (π/sqrt(3))

@assert α2a(a2α(0.5)) ≈ 0.5

# Compute posterior quantiles
begin
	lb(x) = quantile(x, 0.25)
	ub(x) = quantile(x, 0.75)
	llb(x) = quantile(x, 0.025)
	uub(x) = quantile(x, 0.975)
end


# Summarize prior predictive draws relative to true value
function sum_SBC_draws(
	draws;
	params::Vector{Symbol}, # Parameters to sum
	true_values::Vector{Float64},
	prior_var::Vector{Float64} = repeat([1.], length(params)) # Variance of the prior for computing posterior contraction
	)

	if length(size(draws)) > 2 # For MCMC Chains object
		tdraws = DataFrame(Array(draws), names(draws, :parameters))
	else
		tdraws = draws
	end

	sums = []
	for (p, t, pv) in zip(params, true_values, prior_var)

		v = tdraws[!, p]
		v_s = v .- t

		push!(sums,
			(;
				Symbol("$(p)_m") => median(v), # Median posterior
				Symbol("$(p)_lb") => lb(v), # Posterior 25th percentile
				Symbol("$(p)_ub") => ub(v), # Posterior 75th percentile
				Symbol("$(p)_zs") => mean(v_s) / std(v), # Posterior zscore
				Symbol("$(p)_sm") => median(v_s), # Median error
				Symbol("$(p)_slb") => lb(v_s), # Error 25th percentile
				Symbol("$(p)_sub") => ub(v_s), # Error 75th percentile
				Symbol("$(p)_cntrct") => 1 - var(v) / pv, # Posterior contraction, after Schad et al. 2021
				Symbol("true_$p") => t
			)		
		)

	end

	# Return as one NamedTuple
	return reduce((x, y) -> merge(x, y), sums)

end
