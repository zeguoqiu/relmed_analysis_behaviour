function to_standata(
	data::DataFrame,
    initV::Function; # Function that returns initial Q values per block
	model_name::String = "group_QLrs",
    choice_col::Symbol = :isOptimal, # Should be 0, 1
    outcome_col::Symbol = :chosenOutcome,
    PID_col::Symbol = :pp,
    block_col::Symbol = :block
    )

    @assert sort(unique(data[!, choice_col])) == [0, 1]

    cbl = unique(data[!, [PID_col, block_col]])
    cbl.cbl = 1:nrow(cbl)

    cbl = innerjoin(data[!, [PID_col, block_col]], cbl, 
        on = [PID_col, block_col], 
        order = :left)
    
	sd = Dict(
        "N" => nrow(data),
        "N_p" => length(unique(data[!, PID_col])),
		"N_bl" => maximum(data.block),
        "total_N_bl" => nrow(unique(data[!, [PID_col, block_col]])),
		"pp" => data[!, PID_col],
		"bl" => data[!, block_col],
        "cbl" => cbl.cbl,
		"choice" => data[!, choice_col] .+ 0,
		"outcome" => data[!, outcome_col],
		"initV" => initV(data)
    )

    sd["grainsize"] = 1

    # Make sure sorted by PID
    @assert issorted(data[!, PID_col])

    # Pass first row number for each participant
    data_copy = copy(data) # Avoid changing data
    data_copy.rn = 1:nrow(data_copy) # Row numbers
    p1t = combine(groupby(data_copy, PID_col),
        :rn => minimum => :p1t).p1t # First row number per particpant
    push!(p1t, nrow(data_copy)+1) # Add end of dataframe
    sd["p1t"] = p1t

    @assert length(sd["outcome"]) == sd["N"]
    @assert length(sd["choice"]) == sd["N"]
    @assert length(sd["bl"]) == sd["N"]
    @assert length(sd["pp"]) == sd["N"]
	@assert length(sd["p1t"]) == sd["N_p"] + 1

    return sd
end

function load_cmdstanr(
	model_name::String;
	models_dir::String = "./saved_models",
	print_vars::Union{Vector{String}, Missing}=missing,
	cores::Int64=4,
	load_model::Bool = true
)

	# File paths
	model_dir = joinpath(models_dir, model_name) # Subfolder in saved models folder
	draws_file = joinpath(model_dir, "$(model_name)_draws")
	
	# Prepare R script
	r_script = """
	library(cmdstanr)

	fit <- qs::qread("$(joinpath(model_dir, model_name)).qs")

	# Return fit summary
	fit\$summary($(!ismissing(print_vars) ? "variables = c($(join(["\"$var\"" for var in print_vars], ", ")))," : "") .cores = $cores)
	"""

	# Run R script
	if load_model
		fit_summary = RCall.reval(r_script)	
	else
		fit_summary = nothing
	end

	return fit_summary, DataFrame(CSV.File(draws_file))
end

function run_cmdstanr(
	model_name::String,
	model_file::String, # Path to model .stan file
	data::Dict;
	# cmdstan options
	seed::Int64 = 0,
	chains::Int64 = 4,
	parallel_chains::Int64 = chains,
	refresh::Int64 = 100,
	output_dir::String = "./saved_models",
	threads_per_chain::Int64 = 1,
	iter_warmup::Int64 = 1000,
	iter_sampling::Int64 = 1000,
	adapt_delta::Union{Float64, Missing}=missing,
	print_vars::Union{Vector{String}, Missing}=missing
	)

	# File paths
	model_dir = joinpath(output_dir, model_name) # Subfolder in saved models folder
	json_file = joinpath(model_dir, "$(model_name)_data.json") # Data file
	draws_file = joinpath(model_dir, "$(model_name)_draws")
	
	# Create folder for model
	isdir(model_dir) || mkdir(model_dir)

	# Save data to JSON
	open(json_file, "w") do io
	    JSON.print(io, data)
	end

	# Prepare R script
	r_script = """
	library(cmdstanr)
    set_cmdstan_path("/home/jovyan/.cmdstanr/cmdstan-2.34.1")

	# Compile model
	mod <- cmdstan_model(
		"$(model_file)",
		dir = "./tmp",
	    $(threads_per_chain > 1 ? """,
		cpp_options = list(stan_threads = TRUE)""" : "")
	)

	# Fit model
	fit <- mod\$sample(
	  data = "$json_file",
	  seed = $seed,
	  chains = $chains,
	  parallel_chains = $parallel_chains,
	  refresh = $refresh,
	  output_dir = "$model_dir",
	  $(threads_per_chain > 1 ? "threads_per_chain = $threads_per_chain," : "")
	  iter_warmup = $iter_warmup,
	  iter_sampling = $iter_sampling,
	  $(!ismissing(adapt_delta) ? "adapt_delta = $adapt_delta" : "")
	)


	# Save draws as csv
	write.csv(fit\$draws(format = "df"), file = "$draws_file")

	# Save full object for reference
	fit\$draws() # Load posterior draws into the object.
	try(fit\$sampler_diagnostics(), silent = TRUE) # Load sampler diagnostics.
	qs::qsave(x = fit, file = "$(joinpath(model_dir, "$(model_name)")).qs")

	# Return fit summary
	fit\$summary($(!ismissing(print_vars) ? "variables = c($(join(["\"$var\"" for var in print_vars], ", ")))," : "") .cores = $parallel_chains)
	"""
	# Run R script
	start_time = time()
	fit_summary = RCall.reval(r_script)
	end_time = time()
	fit_time = (end_time - start_time) / 60

	return fit_summary, DataFrame(CSV.File(draws_file)), fit_time

end

function load_run_cmdstanr(
	model_name::String,
	model_file::String, # Path to model .stan file
	data::Dict;
	# cmdstan options
	seed::Int64 = 0,
	chains::Int64 = 4,
	parallel_chains::Int64 = chains,
	refresh::Int64 = 100,
	output_dir::String = "./saved_models",
	threads_per_chain::Int64 = 1,
	iter_warmup::Int64 = 1000,
	iter_sampling::Int64 = 1000,
	adapt_delta::Union{Float64, Missing}=missing,
	print_vars::Union{Vector{String}, Missing}=missing,
	load_model::Bool=false # Whehter to load model in R, or just return draws
)
	model_dir = joinpath(output_dir, model_name) # Subfolder in saved models folder
	
	if !isfile("$(joinpath(model_dir, "$(model_name)")).qs")
		println("Fitting model $model_name.")
		
		fit_summary, fit_draws, fit_time = run_cmdstanr(
			model_name,
			"stan_models/$model_file",
			data;
			seed,
			chains,
			parallel_chains,
			refresh,
			output_dir,
			threads_per_chain,
			iter_warmup,
			iter_sampling,
			adapt_delta,
			print_vars
		)
	else
		println("Found previous model fit for $model_name, loading.")
		fit_summary, fit_draws = load_cmdstanr(
			model_name;
			models_dir = output_dir,
			print_vars = print_vars,
			cores = parallel_chains,
			load_model = load_model
		)
		fit_time = missing
	end

	return fit_summary, fit_draws, fit_time

end

# Functions to work with posteriors ----

# Compute posterior quantiles
begin
	lb(x) = quantile(x, 0.25)
	ub(x) = quantile(x, 0.75)
	llb(x) = quantile(x, 0.025)
	uub(x) = quantile(x, 0.975)
end

function sum_p_params(
    draws::DataFrame,
    param::String;
    transform::Bool = true # Wheter to scale and center
)
    # Select columns in draws DataFrame
    tdraws = copy(select(draws, Regex("$(param)\\[\\d+\\]")))

    # Add mean and multiply by SD
    if transform
        tdraws .*= draws[!, Symbol("sigma_$param")]
        tdraws .+= draws[!, Symbol("mu_$param")]	
    end
    
    # Stack
    tdraws = stack(tdraws)	

    # Summarise
    tdraws = combine(groupby(tdraws, :variable),
        :value => median => :median,
        :value => lb => :lb,
        :value => ub => :ub,
        :value => llb => :llb,
        :value => uub => :uub)
    

    # Create :PID columns
    get_pp(s) = parse(Int64, 
        replace(s, Regex("$param\\[(\\d+)\\]") => s"\1"))
    transform!(tdraws,
        :variable => ByRow(get_pp) => :pp)

    return tdraws
end

# This functions summarises the draws from the single_p_QL.stan model
function sum_single_p_QL_draws(single_p_QL_draws::DataFrame,
	true_a::Float64,
	true_ρ::Float64;
	prior_sd::Float64=1.0)

	# Calculate deviations from true value
	a_s = single_p_QL_draws.a .- true_a 
	ρ_s = single_p_QL_draws.rho .- true_ρ

	return (
		a_m = median(single_p_QL_draws.a),
		a_lb = lb(single_p_QL_draws.a),
		a_ub = ub(single_p_QL_draws.a),
		a_zs = mean(a_s) / std(single_p_QL_draws.a),
		a_sm = median(a_s),
		a_slb = lb(a_s),
		a_sub = ub(a_s),
		a_cntrct = 1 - var(single_p_QL_draws.a) / prior_sd ^ 2,
		ρ_m = median(single_p_QL_draws.rho),
		ρ_lb = lb(single_p_QL_draws.rho),
		ρ_ub = ub(single_p_QL_draws.rho),
		ρ_zs = mean(a_s) / std(single_p_QL_draws.rho),
		ρ_sm = median(ρ_s),
		ρ_slb = lb(ρ_s),
		ρ_sub = ub(ρ_s),
		ρ_cntrct = 1 - var(single_p_QL_draws.rho) / prior_sd ^ 2
	)
end

# Summarize prior predictive draws relative to true value
function sum_prior_predictive_draws(
	draws::DataFrame;
	params::Vector{Symbol}, # Parameters to sum
	true_values::Vector{Float64},
	prior_var::Vector{Float64} = repeat([1.], length(params)) # Variance of the prior for computing posterior contraction
	)

	sums = []
	for (p, t, pv) in zip(params, true_values, prior_var)

		v = draws[!, p]
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

# High-level function to simulate a dataset, fit it with stan, and then summarise it.
# Used for prior predictive checks
function simulate_fit_sum(i::Int64;
	n_blocks::Int64 = 10,
	n_trials::Int64 = 13,
	n_participants::Int64 = 1,
	feedback_magnitudes::Vector{Float64} = [1., 2.],
	feedback_ns::Vector{Int64} = [7, 6],
	prior_μ_a::Distribution = Uniform(-2, 2),
	prior_μ_ρ::Distribution = Uniform(0.001, 0.999),
	prior_σ_a::Distribution = Dirac(0.), # For single participant. This just gives 0. always
	prior_σ_ρ::Distribution = Dirac(0.),
	model::String = "single_p_QL",
	name::String = ""
	)

	# Draw hyper-parameters
	true_μ_a = rand(prior_μ_a)
	true_μ_ρ = abs(rand(prior_μ_ρ))
	true_σ_a = abs(rand(prior_σ_a))
	true_σ_ρ = abs(rand(prior_σ_ρ))
	
	# Simulate data
	sim_dat = simulate_q_learning_dataset(
		n_participants,
		n_trials,
		repeat([feedback_magnitudes], n_blocks),
		repeat([feedback_ns], n_blocks),
		vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-2]],
			repeat([[n_trials-3]], n_blocks - 4)),
		true_μ_a,
		true_σ_a,
		true_μ_ρ,
		true_σ_ρ;
		random_seed = i)

	sim_dat.isOptimal = (sim_dat.choice .== 1) .+ 0

	# Fit model
	_, QL_draws = load_run_cmdstanr(
		"sim_$(model)_$(name != "" ? name * "_" : "")$(i)",
		"$model.stan",
		to_standata(sim_dat,
			x -> repeat([sum(feedback_magnitudes .* feedback_ns) / 
			(sum(feedback_ns) * 2)], 2);
			PID_col = :PID,
			outcome_col = :outcome,
			model_name = model);
		threads_per_chain = occursin("rs", model) ? 3 : 1
	)

	# Summarize draws
	var_half_normal = var(truncated(Normal(), lower = 0))

	sum_draws = sum_prior_predictive_draws(QL_draws,
		params = n_participants == 1 ? [:a, :rho] : [:mu_a, :mu_rho, :sigma_a, :sigma_rho, Symbol("a[1]"), Symbol("rho[1]")],
		true_values = n_participants == 1 ? [true_μ_a, true_μ_ρ] : [true_μ_a, true_μ_ρ, true_σ_a, true_σ_ρ, 
			quantile(Normal(), filter(x -> x.PID == 1, sim_dat).α[1]),
			filter(x -> x.PID == 1, sim_dat).ρ[1]],
		prior_var = n_participants == 1 ? [1., 1.] : [1., 1., var_half_normal, var_half_normal, 1., 1.]
	)

	return (; sum_draws..., 
		n_blocks = n_blocks,
		n_trials = n_trials)
	
end