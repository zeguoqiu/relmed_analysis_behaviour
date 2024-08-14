### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ fe776a2c-589a-11ef-0331-472f2769c12f
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf
	using LogExpFunctions: logistic, logit

	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/independent_group_QL.jl")
end

# ╔═╡ 998eed23-55e1-4bcd-b995-21ca82577454
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	th = Theme(
		font = "Helvetica",
		fontsize = 16,
		Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 14,
			yticklabelsize = 14,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
		)
	)
	set_theme!(th)
end

# ╔═╡ ba3495da-93cc-4002-a2a5-99589a819a94
# Simulate one dataset
prior_sample = let n_participants = 10
	
	# Load sequence from file
	task = DataFrame(CSV.File("data/PLT_task_structure_00.csv"))

	# Renumber block
	task.block = task.block .+ (task.session .- 1) * maximum(task.block)

	# Arrange feedback by optimal / suboptimal
	task.feedback_optimal = 
		ifelse.(task.optimal_A .== 1, task.feedback_A, task.feedback_B)

	task.feedback_suboptimal = 
		ifelse.(task.optimal_A .== 0, task.feedback_A, task.feedback_B)


	# Arrange outcomes such as second column is optimal
	outcomes = hcat(
		task.feedback_suboptimal,
		task.feedback_optimal,
	)

	# Arrange values for simulation
	participant = repeat(1:n_participants, inner = length(task.block))
	block = repeat(task.block, n_participants)
	valence = repeat(task.valence, n_participants)
	outcomes = repeat(outcomes, n_participants)

	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	simulate_independent_group_QL(
		1;
		participant = participant,
		block = block,
		valence = valence,
		outcomes = outcomes,
		initV = fill(aao, 1, 2),
		random_seed = 0
	)

end

# ╔═╡ 76bf001d-c20f-4c23-bc35-a483679a96e0
# Plot prior preditctive accuracy curve
let
	# Unpack prior sample into DataFrame
	df = DataFrame(
		:PID => prior_sample[:participant],
		:block => prior_sample[:block],
		:valence => prior_sample[:valence],
		:choice => prior_sample[:choice],
		:EV_A => prior_sample[:Qs][:, 2],
		:EV_B => prior_sample[:Qs][:, 1],
		:ρ => prior_sample[:ρ][prior_sample[:participant]]
	)

	# Add trial column
	DataFrames.transform!(
		groupby(df, [:PID, :block]),
		:block => (x -> 1:length(x)) => :trial
	)

	df[!, :group] .= 1

	f = Figure(size = (700, 1000))

	g_all = f[1,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_all,
		df;
		plw = 1,
		legend = false,
		acc_error_band = "PI"
	)

	g_reward = f[2,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_reward,
		filter(x -> x.valence > 0, df);
		plw = 1,
		legend = false,
		acc_error_band = "PI"
	)

	g_reward = f[3,1] = GridLayout()
	
	plot_sim_q_value_acc!(
		g_reward,
		filter(x -> x.valence < 0, df);
		plw = 1,
		legend = false,
		acc_error_band = "PI"
	)

	f

end

# ╔═╡ 5f3fea09-f7a5-4f2c-89f3-091c83f1abe8
let

	fit = optimize_independent_group_QL(;
		participant = prior_sample[:participant],
		block = prior_sample[:block],
		valence = prior_sample[:valence],
		choice = prior_sample[:choice],
		outcomes = prior_sample[:outcomes],
		initV = prior_sample[:initV],
		estimate = "MAP"
	)

	

end

# ╔═╡ 32b338c0-3a04-4759-bff0-8c04a3201f56
loglikelihood(independent_group_QL_flat(;
	N = length(prior_sample[:block]),
	n_p = maximum(prior_sample[:participant]),
	participant = prior_sample[:participant],
	block = prior_sample[:block],
	valence = prior_sample[:valence],
	choice = prior_sample[:choice],
	outcomes = prior_sample[:outcomes],
	initV = prior_sample[:initV]
), (ρ = fill(1., 50), a = fill(1., 50)))

# ╔═╡ Cell order:
# ╠═fe776a2c-589a-11ef-0331-472f2769c12f
# ╠═998eed23-55e1-4bcd-b995-21ca82577454
# ╠═ba3495da-93cc-4002-a2a5-99589a819a94
# ╠═76bf001d-c20f-4c23-bc35-a483679a96e0
# ╠═5f3fea09-f7a5-4f2c-89f3-091c83f1abe8
# ╠═32b338c0-3a04-4759-bff0-8c04a3201f56
