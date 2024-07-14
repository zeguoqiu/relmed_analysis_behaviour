function to_standata(
	sim_dat::DataFrame,
    feedback_magnitudes::Vector{Float64},
    feedback_ns::Vector{Int64};
	model_name::String = "single_QL",
    )
    
	sd = Dict(
        "N" => nrow(sim_dat),
        "N_p" => length(unique(sim_dat.PID)),
		"N_bl" => maximum(sim_dat.block),
		"pp" => sim_dat.PID,
		"bl" => sim_dat.block,
		"choice" => (sim_dat.choice .== 2) .+ 0,
		"outcome" => sim_dat.outcome,
		"initV" => repeat([sum(feedback_magnitudes .* feedback_ns) / 
			(sum(feedback_ns) * 2)], 2)
    )

	if model_name == "group_QLrs"
		sd["grainsize"] = 1

		# Make sure sorted by PID
		all(sim_dat.PID[i] <= sim_dat.PID[i+1] for i in 1:length(sim_dat.PID)-1)

		# Pass first row number for each participant
		sim_dat_copy = copy(sim_dat) # Avoid changing sim_dat
		sim_dat_copy.rn = 1:nrow(sim_dat_copy) # Row numbers
		p1t = combine(groupby(sim_dat_copy, :PID),
			:rn => minimum => :p1t).p1t # First row number per particpant
		push!(p1t, nrow(sim_dat_copy)+1) # Add end of dataframe
		sd["p1t"] = p1t
	end
	
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
			model_file,
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