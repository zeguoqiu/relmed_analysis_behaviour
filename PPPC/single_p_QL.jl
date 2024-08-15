### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ fb94ad20-57e0-11ef-2dae-b16d3d00e329
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
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
end

# ╔═╡ 261d0d08-10b9-4111-9fc8-bb84e6b4cef5
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

# ╔═╡ fa79c576-d4c9-4c42-b5df-66d886e8abe4
# Sample datasets from prior
begin
	prior_sample = let
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
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_sample = simulate_single_p_QL(
			200;
			block = task.block,
			valence = unique(task[!, [:block, :valence]]).valence,
			outcomes = outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0
		)
	
	
		leftjoin(prior_sample, 
			task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_sample)
end

# ╔═╡ d7b60f28-09b1-42c0-8c95-0213590d8c5c
# Plot prior preditctive accuracy curve
let
	df = rename(prior_sample, 
		:Q_optimal => :EV_A,
		:Q_suboptimal => :EV_B
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

# ╔═╡ f5113e3e-3bcf-4a92-9e76-d5eed8088320
md"""
## Sampling from posterior
"""

# ╔═╡ 9477b295-ada5-46cf-b2e3-2c1303873081
# Sample from posterior and plot for single participant
begin
	fit = let
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		
		fit = posterior_sample_single_p_QL(
			filter(x -> x.PID == 1, prior_sample); 
			initV = aao,
			random_seed = 0
		)
	
		fit
	
	end

	f_one_posterior = plot_posteriors([fit],
		["a", "ρ"];
		true_values = [α2a(prior_sample[1, :α]), prior_sample[1, :ρ]]
	)	

	ax_cor = Axis(
		f_one_posterior[1,3],
		xlabel = "a",
		ylabel = "ρ",
		aspect = 1,
		xticks = WilkinsonTicks(4)
	)

	scatter!(
		ax_cor,
		fit[:, :a, :] |> vec,
		fit[:, :ρ, :] |> vec,
		markersize = 1.5
	)

	colsize!(f_one_posterior.layout, 3, Relative(0.2))

	f_one_posterior
end

# ╔═╡ 2afaba84-49b6-4770-a3bb-6e8e4c8be4ba
sbc = let
	draws_sum_file = "saved_models/sbc_single_p_QL.jld2"
	if isfile(draws_sum_file)
		JLD2.@load draws_sum_file sbc
	else
		sbc = SBC_single_p_QL(
			filter(x -> x.PID <= 100, prior_sample);
			initV = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]),
			random_seed = 0
		) |> DataFrame

		JLD2.@save draws_sum_file sbc
	end

	sbc
end

# ╔═╡ 50d88c5c-4a3f-4ded-81cf-600eddb3bbf9
let
	f_SBC = plot_SBC(sbc, show_n = [1], params = ["a", "ρ"])

	resize!(f_SBC.scene, (700,700))

	ax_cor1 = Axis(
		f_SBC[3,1],
		xlabel = "Posterior estimate of a",
		ylabel = "Posterior estimate of ρ"
	)

	scatter!(
		ax_cor1,
		sbc.a_m,
		sbc.ρ_m,
		markersize = 4
	)

	ax_cor2 = Axis(
		f_SBC[3,2],
		xlabel = "True a",
		ylabel = "True ρ"
	)

	scatter!(
		ax_cor2,
		sbc.true_a,
		sbc.true_ρ,
		markersize = 4
	)


	f_SBC

end

# ╔═╡ 47b4f578-98ee-4ae0-8359-f4e8de5a63f1
md"""
## Estimation by optimization
"""

# ╔═╡ bcb0ff89-a02f-43b7-9015-f7c3293bc2ec
function optimization_calibration(
	prior_sample;
	estimate::String = "MLE",
	ms::Float64 = 4.
)
	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	MLEs = optimize_multiple_single(
		prior_sample;
		initV = aao,
		estimate = estimate,
		include_true = true
	)

	f = Figure(size = (700, 200))

	# Plot a
	ax_a = Axis(
		f[1,1],
		xlabel = "True a",
		ylabel = "$estimate a",
		aspect = 1.
	)

	scatter!(
		ax_a,
		MLEs.true_a,
		MLEs.MLE_a,
		markersize = ms
	)

	unit_line!(ax_a)

	# Plot ρ
	ax_ρ = Axis(
		f[1,2],
		xlabel = "True ρ",
		ylabel = "$estimate ρ",
		aspect = 1.
	)

	scatter!(
		ax_ρ,
		MLEs.true_ρ,
		MLEs.MLE_ρ,
		markersize = ms
	)

	unit_line!(ax_ρ)

	# Plot bivariate
	ax_aρ = Axis(
		f[1,3],
		xlabel = "$estimate a",
		ylabel = "$estimate ρ",
		aspect = 1.
	)

	scatter!(
		ax_aρ,
		MLEs.MLE_a,
		MLEs.MLE_ρ,
		markersize = ms
	)

	# Plot ground truth
	ax_taρ = Axis(
		f[1,4],
		xlabel = "True a",
		ylabel = "True ρ",
		aspect = 1.
	)

	scatter!(
		ax_taρ,
		MLEs.true_a,
		MLEs.true_ρ,
		markersize = ms
	)

	f
end

# ╔═╡ 1d982252-c9ac-4925-9cd5-976456d32bc4
optimization_calibration(
	prior_sample,
	estimate = "MLE"
)

# ╔═╡ 2eb2dd61-abae-4328-9787-7a841d321836
optimization_calibration(
	prior_sample;
	estimate = "MAP"
)

# ╔═╡ a3c8a90e-d820-4542-9043-e06a0ec9eaee
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ e7eac420-5048-4496-a9bb-04eca8271b17
function prepare_for_fit(data)

	forfit = select(data, [:prolific_pid, :session, :block, :valence, :trial, :optimalRight, :outcomeLeft, :outcomeRight, :chosenOutcome, :isOptimal])

	rename!(forfit, :isOptimal => :choice)

	# Arrange feedback by optimal / suboptimal
	forfit.feedback_optimal = 
		ifelse.(forfit.optimalRight .== 1, forfit.outcomeRight, forfit.outcomeLeft)

	forfit.feedback_suboptimal = 
		ifelse.(forfit.optimalRight .== 0, forfit.outcomeRight, forfit.outcomeLeft)

	# PID as number
	pids = DataFrame(
		prolific_pid = unique(forfit.prolific_pid)
	)

	pids.PID = 1:nrow(pids)

	forfit = innerjoin(forfit, pids, on = :prolific_pid)

	# Block as Int64
	forfit.block = convert(Vector{Int64}, forfit.block)

	return forfit, pids
end

# ╔═╡ e6639a00-8135-482b-88bc-de2a8a8a4a94
function reliability_scatter(
	fit1::DataFrame,
	fit2::DataFrame,
	pids1::DataFrame,
	pids2::DataFrame,
	label1::String,
	label2::String
)

	# Join
	fit1 = innerjoin(fit1, pids1, on = :PID)
	
	fit2 = innerjoin(fit2, pids2, on = :PID)
	
	fits = innerjoin(
		fit1[!, Not(:PID)], 
		fit2[!, Not(:PID)], 
		on = [:prolific_pid],
		renamecols = "_1" => "_2"
	)

	# Plot -----------------------------------
	f = Figure()

	scatter_regression_line!(
		f[1,1],
		fits,
		:a_1,
		:a_2,
		"$label1 a",
		"$label2 a"
	)

	scatter_regression_line!(
		f[1,2],
		fits,
		:ρ_1,
		:ρ_2,
		"$label1 ρ",
		"$label2 ρ"
	)


	return f
end


# ╔═╡ cc82e036-f83c-4f33-847a-49f3a3ec9342
# Test-retest
let

	sess1_forfit, sess1_pids = prepare_for_fit(filter(x -> x.session == "1", PLT_data))
	sess2_forfit, sess2_pids = prepare_for_fit(filter(x -> x.session == "2", PLT_data))
	
	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	# Fit
	sess1_maps = optimize_multiple_single(
		sess1_forfit;
		initV = aao,
	)

	sess2_maps = optimize_multiple_single(
		sess2_forfit;
		initV = aao,
	)

	reliability_scatter(
		sess1_maps,
		sess2_maps,
		sess1_pids,
		sess2_pids,
		"Session 1",
		"Session 2"
	)

end

# ╔═╡ Cell order:
# ╠═fb94ad20-57e0-11ef-2dae-b16d3d00e329
# ╠═261d0d08-10b9-4111-9fc8-bb84e6b4cef5
# ╠═fa79c576-d4c9-4c42-b5df-66d886e8abe4
# ╠═d7b60f28-09b1-42c0-8c95-0213590d8c5c
# ╟─f5113e3e-3bcf-4a92-9e76-d5eed8088320
# ╠═9477b295-ada5-46cf-b2e3-2c1303873081
# ╠═2afaba84-49b6-4770-a3bb-6e8e4c8be4ba
# ╠═50d88c5c-4a3f-4ded-81cf-600eddb3bbf9
# ╟─47b4f578-98ee-4ae0-8359-f4e8de5a63f1
# ╠═1d982252-c9ac-4925-9cd5-976456d32bc4
# ╠═2eb2dd61-abae-4328-9787-7a841d321836
# ╠═bcb0ff89-a02f-43b7-9015-f7c3293bc2ec
# ╠═a3c8a90e-d820-4542-9043-e06a0ec9eaee
# ╠═e7eac420-5048-4496-a9bb-04eca8271b17
# ╠═e6639a00-8135-482b-88bc-de2a8a8a4a94
# ╠═cc82e036-f83c-4f33-847a-49f3a3ec9342
