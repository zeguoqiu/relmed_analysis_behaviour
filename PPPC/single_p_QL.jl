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

	MLEs = optimize_multiple_single_p_QL(
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

	DataFrames.transform!(groupby(PLT_data, [:prolific_pid, :session, :block]),
		:isOptimal => count_consecutive_ones => :consecutiveOptimal
	)

	PLT_data = exclude_PLT_trials(PLT_data)

	nothing
end

# ╔═╡ cc82e036-f83c-4f33-847a-49f3a3ec9342
# Test-retest
function reliability_by_condition(
	PLT_data::DataFrame,
	filter1::Function,
	filter2::Function,
	label_top::String,
	label_bottom1::String,
	label_bottom2::String;
	supertitle::String = ""
)
	forfit1, pids1 = 
		prepare_for_fit(filter(filter1, PLT_data))
	forfit2, pids2 = 
		prepare_for_fit(filter(filter2, PLT_data))
	
	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	# Fit
	maps1 = optimize_multiple_single(
		forfit1;
		initV = aao,
		σ_ρ = 1.,
		σ_a = 0.5
	)

	maps2 = optimize_multiple_single(
		forfit2;
		initV = aao,
		σ_ρ = 1.,
		σ_a = 0.5
	)

	# Join
	maps = join_split_fits(
		maps1,
		maps2,
		pids1,
		pids2
	)

	# Add condition data
	maps = innerjoin(
		maps, 
		unique(
			PLT_data[!, [:prolific_pid, :early_stop, :valence_grouped, :reward_first]]
		),
		on = :prolific_pid
	)

	maps.reward_first = ifelse.(maps.valence_grouped, maps.reward_first, missing)

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

	bootstrap_correlation(maps.a_1, maps.a_2)

	# Compute correlations
	function groupby_cor(
		maps::DataFrame,
		col::Symbol,
		label::String
	)
		cors = combine(
			groupby(maps, col),
			[:a_1, :a_2] => bootstrap_correlation => :cor_a,
			[:ρ_1, :ρ_2] => bootstrap_correlation => :cor_ρ,
			:prolific_pid => length => :n
		)

		cors[!, :variable] .= label

		rename!(cors, col => :level)

		return cors
	end

	cat_cors = vcat(
		groupby_cor(maps, :early_stop, "Early stopping"),
		groupby_cor(maps, :reward_first, "Reward first")
	)

	cat_cors = innerjoin(
		cat_cors,
		DataFrame(
			variable = unique(cat_cors.variable),
			cat = 1:length(unique(cat_cors.variable)),
		),
		on = :variable
	)

	cat_cors = innerjoin(
		cat_cors,
		DataFrame(
			level = unique(cat_cors.level),
			level_id = 1:length(unique(cat_cors.level)),
		),
		on = :level,
		matchmissing = :equal
	)

	f = Figure(size = (1000,500))

	f_top = f[1,1] = GridLayout()

	function plot_cor_dist(
		f::GridPosition, 
		cat_cors::DataFrame, 
		col::Symbol;
		ylabel::String = "",
		title::String
	)
		
		ax = Axis(
			f,
			xticks = (1:2, unique(cat_cors.variable)),
			ylabel = ylabel,
			title = title
		)
	
		rainclouds!(
			ax,
			cat_cors.variable,
			cat_cors[!, col],
			dodge = cat_cors.level_id,
			color = Makie.wong_colors()[cat_cors.level_id],
			plot_boxplots = false
		)
	
	end

	plot_cor_dist(f[1,1], cat_cors, :cor_a; ylabel = label_top, title = "a")
	plot_cor_dist(f[1,2], cat_cors, :cor_ρ; ylabel = label_top, title = "ρ")
	
	Legend(
		f[1,1],
		[PolyElement(color = c) for c in Makie.wong_colors()[1:3]],
		["No", "Yes", "N/A"],
		orientation = :horizontal,
		framevisible = false,
		tellwidth = false,
		tellheight = false,
		halign = :right,
		valign = :bottom
	)

	if supertitle != ""
		Label(f[0, :], supertitle, fontsize = 18, font = :bold)
	end

	# Plot scatter

	gl1 = f[2,1:2] = GridLayout()

	ax_a1 = Axis(
		gl1[1,1],
		xlabel = "$label_bottom1 a",
		ylabel = "$label_bottom2 a",
	)

	ax_a2 = Axis(
		gl1[1,2],
		xlabel = "$label_bottom1 a",
		ylabel = "$label_bottom2 a",
	)

	ax_ρ1 = Axis(gl1[1,3],
		xlabel = "$label_bottom1 ρ",
		ylabel = "$label_bottom2 ρ",
	)

	ax_ρ2 = Axis(gl1[1,4],
		xlabel = "$label_bottom1 ρ",
		ylabel = "$label_bottom2 ρ",
	)

	for (i, es) in enumerate(unique(maps.early_stop))	
		reliability_scatter!(
			ax_a1,
			ax_ρ1,
			filter(x -> x.early_stop == es, maps),
			"Session 1",
			"Session 2",
			color = Makie.wong_colors()[i]
		)
	end

	for (i, rf) in enumerate(unique(maps.reward_first))	
		reliability_scatter!(
			ax_a2,
			ax_ρ2,
			filter(x -> isequal(x.reward_first, rf), maps),
			"Session 1",
			"Session 2",
			color = Makie.wong_colors()[i]
		)
	end


	f

end

# ╔═╡ 0a151f69-f59e-48ae-8fc2-46a455e4f049
reliability_by_condition(
	PLT_data,
	x -> x.session == "1",
	x -> x.session == "2",
	"Test-retest",
	"Session 1",
	"Session 2"
)

# ╔═╡ fff23ca1-35bc-4ff1-aea6-d9bb5ce86b1f
reliability_by_condition(
	PLT_data,
	x -> (x.session == "1") & iseven(x.block),
	x -> (x.session == "1") & isodd(x.block),
	"Split-half",
	"Even blocks",
	"Odd block";
	supertitle = "Session 1 split-half reliability"
)

# ╔═╡ 8c9a0662-25af-4280-ad48-270458edb018
reliability_by_condition(
	PLT_data,
	x -> (x.session == "2") & iseven(x.block),
	x -> (x.session == "2") & isodd(x.block),
	"Split-half",
	"Even blocks",
	"Odd block";
	supertitle = "Session 2 split-half reliability"
)

# ╔═╡ 525522d1-5ced-46f2-8c9b-3299d3cb244d
begin
	find_excess = x -> (x.consecutiveOptimal <= 5) | 
		((x.consecutiveOptimal == 6) && (x.trial == 13))
	
	test = filter(x -> !find_excess(x), PLT_data)

	@assert all([s[1] == '0' for s in unique(test.condition)])

	shortened_PLT = filter(find_excess, PLT_data)
end

# ╔═╡ 479726b5-605b-494e-8ff2-4d569d0c0ddd
reliability_by_condition(
	shortened_PLT,
	x -> x.session == "1",
	x -> x.session == "2",
	"Test-retest",
	"Session 1",
	"Session 2";
	supertitle = "Early stopping simulated for all"
)

# ╔═╡ 3a8f569d-0d8e-4020-9023-a123cad9d5de
reliability_by_condition(
	shortened_PLT,
	x -> (x.session == "1") & iseven(x.block),
	x -> (x.session == "1") & isodd(x.block),
	"Split-half",
	"Even blocks",
	"Odd block";
	supertitle = "Session 1 split-half reliability\nearly stopping simulated for all"
)

# ╔═╡ 10565355-90ae-419c-9b92-8ff18fcd48b3
reliability_by_condition(
	shortened_PLT,
	x -> (x.session == "2") & iseven(x.block),
	x -> (x.session == "2") & isodd(x.block),
	"Split-half",
	"Even blocks",
	"Odd block";
	supertitle = "Session 1 split-half reliability\nearly stopping simulated for all"
)

# ╔═╡ a53db393-c9e7-4db3-a11d-3b244823d951
# Different priors
penlaties_fits = let
	sess1_forfit, sess1_pids = 
		prepare_for_fit(filter(x -> x.session == "1", PLT_data))
	sess2_forfit, sess2_pids = 
		prepare_for_fit(filter(x -> x.session == "2", PLT_data))
	
	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	function fit_split_compute_cor(
		data1::DataFrame,
		data2::DataFrame,
		pids1::DataFrame,
		pids2::DataFrame;
		σ_ρ::Float64,
		σ_a::Float64,
		ids::NamedTuple = (σ_ρ = σ_ρ, σ_a = σ_a)
	)
		# Fit
		fit1 = optimize_multiple_single(
			data1;
			initV = aao,
			σ_ρ = σ_ρ
		)
	
		fit2 = optimize_multiple_single(
			data2;
			initV = aao,
			σ_ρ = σ_a
		)
	
		maps = join_split_fits(
			fit1,
			fit2,
			pids1,
			pids2
		)
	
		return merge(
			ids, 
			(
				cor_a = cor(maps.a_1, maps.a_2),
				cor_ρ = cor(maps.ρ_1, maps.ρ_2)
			)
		)
	end

	σ_as = range(0.5, 20., length = 10)
	σ_ρs = range(1., 20., length = 10)

	fits = [fit_split_compute_cor(
		sess1_forfit, 
		sess2_forfit, 
		sess1_pids, 
		sess2_pids;
		σ_ρ = σ_ρ,
		σ_a = σ_a
		) for σ_ρ in σ_ρs for σ_a in σ_as
	]
	
	fits = DataFrame(fits)

end

# ╔═╡ a0c99fd5-38fa-4117-be5d-c3eb7fd0ce5f
best_penalties = let 
	penlaties_fits.sum_r_sq = penlaties_fits.cor_a .^ 2 + penlaties_fits.cor_ρ .^2

	max_fit = filter(x -> x.sum_r_sq == maximum(penlaties_fits.sum_r_sq), penlaties_fits)

end

# ╔═╡ a88ffe29-f0f4-4eb7-8fc3-a7fcc08560d0
let

	f = Figure()

	joint_limits = (
		minimum(Array(penlaties_fits[!, [:cor_a, :cor_ρ]])),
		maximum(Array(penlaties_fits[!, [:cor_a, :cor_ρ]]))
	)


	ax_a = Axis(
		f[1,1],
		xlabel = "σ_a",
		ylabel = "σ_ρ",
		aspect = 1.,
		title = "Test retest for a"
	)

	heatmap!(
		ax_a,
		penlaties_fits.σ_a,
		penlaties_fits.σ_ρ,
		penlaties_fits.cor_a,
		colorrange = joint_limits
	)

	ax_ρ = Axis(
		f[1,2],
		xlabel = "σ_a",
		ylabel = "σ_ρ",
		aspect = 1.,
		title = "Test retest for ρ"
	)

	hm = heatmap!(
		ax_ρ,
		penlaties_fits.σ_a,
		penlaties_fits.σ_ρ,
		penlaties_fits.cor_ρ,
		colorrange = joint_limits
	)

	Colorbar(f[:, end+1], hm, tellheight = true)

	rowsize!(f.layout, 1, ax_a.scene.viewport[].widths[2])

	f

	

	
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
# ╠═0a151f69-f59e-48ae-8fc2-46a455e4f049
# ╠═fff23ca1-35bc-4ff1-aea6-d9bb5ce86b1f
# ╠═8c9a0662-25af-4280-ad48-270458edb018
# ╠═cc82e036-f83c-4f33-847a-49f3a3ec9342
# ╠═525522d1-5ced-46f2-8c9b-3299d3cb244d
# ╠═479726b5-605b-494e-8ff2-4d569d0c0ddd
# ╠═3a8f569d-0d8e-4020-9023-a123cad9d5de
# ╠═10565355-90ae-419c-9b92-8ff18fcd48b3
# ╠═a53db393-c9e7-4db3-a11d-3b244823d951
# ╠═a0c99fd5-38fa-4117-be5d-c3eb7fd0ce5f
# ╠═a88ffe29-f0f4-4eb7-8fc3-a7fcc08560d0
