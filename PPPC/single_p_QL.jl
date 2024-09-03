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
    Pkg.instantiate()
	using CairoMakie, Random, DataFrames, Distributions, StatsBase,
		ForwardDiff, LinearAlgebra, JLD2, FileIO, CSV, Dates, JSON, RCall, Turing, ParetoSmooth, MCMCDiagnosticTools, Printf
	using LogExpFunctions: logistic, logit

	include("$(pwd())/PILT_models.jl")
	include("$(pwd())/sample_utils.jl")
	include("$(pwd())/stats_utils.jl")
	include("$(pwd())/plotting_utils.jl")
	include("$(pwd())/single_p_QL.jl")
	include("$(pwd())/fetch_preprocess_data.jl")
	nothing
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
		task = task_vars_for_condition("00")
	
		# Initial value for Q values
		aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
	
		prior_sample = simulate_single_p_QL(
			200;
			block = task.block,
			valence = task.valence,
			outcomes = task.outcomes,
			initV = fill(aao, 1, 2),
			random_seed = 0,
			prior_ρ = truncated(Normal(0., 2.), lower = 0.),
			prior_a = Normal()
		)
	
	
		leftjoin(prior_sample, 
			task.task[!, [:block, :trial, :feedback_optimal, :feedback_suboptimal]],
			on = [:block, :trial]
		)
	
	end

	describe(prior_sample)
end

# ╔═╡ 069177a1-d078-4d3c-bfa3-1bef04065d89
let

	f = plot_prior_predictive_by_valence(
		prior_sample,
		[:Q_optimal, :Q_suboptimal]
	)

	save("results/single_p_QL_prior_predictive.png", f, pt_per_unit = 1)

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

	save("results/single_p_QL_example_posterior.png", f_one_posterior, pt_per_unit = 1)

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


	save("results/single_p_QL_sampling_calibration.png", f_SBC, pt_per_unit = 1)

	f_SBC

end

# ╔═╡ 47b4f578-98ee-4ae0-8359-f4e8de5a63f1
md"""
## Estimation by optimization
"""

# ╔═╡ 1d982252-c9ac-4925-9cd5-976456d32bc4
let
	f = optimization_calibration(
		prior_sample,
		optimize_multiple_single_p_QL,
		estimate = "MLE",
		prior_ρ = missing,
		prior_a = missing
	)

	save("results/single_p_QL_MLE_calibration.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ 2eb2dd61-abae-4328-9787-7a841d321836
let
	f = optimization_calibration(
		prior_sample,
		optimize_multiple_single_p_QL;
		estimate = "MAP",
		prior_ρ = truncated(Normal(0., 2.), lower = 0.),
		prior_a = Normal()
	)

	save("results/single_p_QL_PMLE_calibration.png", f, pt_per_unit = 1)

	f
end

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

# ╔═╡ 525522d1-5ced-46f2-8c9b-3299d3cb244d
begin
	find_excess = x -> (x.consecutiveOptimal <= 5) | 
		((x.consecutiveOptimal == 6) && (x.trial == 13))
	
	test = filter(x -> !find_excess(x), PLT_data)

	@assert all([s[1] == '0' for s in unique(test.condition)])

	shortened_PLT = filter(find_excess, PLT_data)
end

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
		fit1 = optimize_multiple_single_p_QL(
			data1;
			initV = aao,
			prior_ρ = truncated(Normal(0., σ_ρ), lower = 0.),
			prior_a = Normal(0., σ_a)
		)
	
		fit2 = optimize_multiple_single_p_QL(
			data2;
			initV = aao,
			prior_ρ = truncated(Normal(0., σ_ρ), lower = 0.),
			prior_a = Normal(0., σ_a)
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

	
	save("results/single_p_QL_PMLE_test_retest_by_prior.png", f, 
		pt_per_unit = 1)

	f

	
end

# ╔═╡ 8ebaeb8e-e760-4f9e-a6bc-e3ecf44e6665
best_penalties = let
       penlaties_fits.avg_r_sq = 
		   (penlaties_fits.cor_a .^ 2 + penlaties_fits.cor_ρ .^2) / 2

       max_fit = 
		   filter(x -> x.avg_r_sq == maximum(penlaties_fits.avg_r_sq), 
			   penlaties_fits)
end

# ╔═╡ c6558729-ed5b-440b-8e59-e69071b26f09
aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

# ╔═╡ 5ebdc7b1-1c94-425d-9e08-0fc7da8cdc34
function bootstrap_fit(
	PLT_data::AbstractDataFrame;
	n_bootstraps::Int64 = 20,
	initV::Float64 = aao,
	prior_ρ::Distribution = truncated(Normal(0., 2.), lower = 0.),
	prior_a::Distribution = Normal()
) 
	forfit, pids = prepare_for_fit(PLT_data)

	fit = optimize_multiple_single_p_QL(
		forfit;
		initV = initV,
		estimate = "MAP",
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	bootstraps = vcat([insertcols(
		fit[sample(Xoshiro(i), 1:nrow(fit), nrow(fit), replace=true), :],
		:bootstrap_idx => i
	) for i in 1:n_bootstraps]...)

	return bootstraps

end

# ╔═╡ de41a8c1-fc09-4c33-b371-4d835a0a46ce
function fit_split(
	PLT_data::DataFrame,
	filter1::Function,
	filter2::Function,
	prior_ρ::Union{Distribution, Missing} = truncated(Normal(0., 2.), lower = 0.),
	prior_a::Union{Distribution, Missing} = Normal()

)
	forfit1, pids1 = 
		prepare_for_fit(filter(filter1, PLT_data))
	forfit2, pids2 = 
		prepare_for_fit(filter(filter2, PLT_data))
	
	# Initial value for Q values
	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])

	# Fit
	maps1 = optimize_multiple_single_p_QL(
		forfit1;
		initV = aao,
		prior_ρ = prior_ρ,
		prior_a = prior_a
	)

	maps2 = optimize_multiple_single_p_QL(
		forfit2;
		initV = aao,
		prior_ρ = prior_ρ,
		prior_a = prior_a
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

end

# ╔═╡ c2e51103-b6ee-43a3-87e5-a9a2b28ed8e4
# Overall test-retest
let

	maps = fit_split(PLT_data, x -> x.session == "1", x -> x.session == "2")
	
	f = reliability_scatter(
		maps,
		"Session 1",
		"Session 2"
	)

	Label(f[0,:], "Test-retest reliability", fontsize = 18, font = :bold, 
		tellwidth = false)

	rowsize!(f.layout, 0, Relative(0.1))

	save("results/single_p_QL_PMLE_test_retest.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 48343b75-a0cd-4806-9867-b861b118491d
# Overall split-half
let

	maps = fit_split(PLT_data, x -> (x.session == "1") & isodd(x.block),
		x -> (x.session == "1") & iseven(x.block))
	
	f = reliability_scatter(
		maps,
		"Odd blocks",
		"Even blocks"
	)

	Label(f[0,:], "Session 1 split-half correlation", fontsize = 18, font = :bold, 
		tellwidth = false)

	rowsize!(f.layout, 0, Relative(0.1))

	save("results/single_p_QL_PMLE_split_half_sess1.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 8a567ab1-db8a-47df-ac2b-59714f230ae6
# Overall split-half session 2
let

	maps = fit_split(PLT_data, x -> (x.session == "2") & isodd(x.block),
		x -> (x.session == "2") & iseven(x.block))
	
	f = reliability_scatter(
		maps,
		"Odd blocks",
		"Even blocks"
	)

	Label(f[0,:], "Session 2 split-half correlation", fontsize = 18, font = :bold, 
		tellwidth = false)

	rowsize!(f.layout, 0, Relative(0.1))

	save("results/single_p_QL_PMLE_split_half_sess2.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 2239dd1c-1975-46b4-b270-573efd454c04
function reliability_by_condition(
	PLT_data::DataFrame,
	filter1::Function,
	filter2::Function,
	label_top::String,
	label_bottom1::String,
	label_bottom2::String;
	supertitle::String = ""
)
	maps = fit_split(PLT_data, filter1, filter2)

	maps.reward_first = ifelse.(maps.valence_grouped, maps.reward_first, missing)

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

	reward_first_levels = unique(cat_cors.level)
	cat_cors = innerjoin(
		cat_cors,
		DataFrame(
			level = unique(cat_cors.level),
			level_id = 1:length(reward_first_levels),
		),
		on = :level,
		matchmissing = :equal
	)

	f = Figure(size = (1000,500))

	f_top = f[1,1] = GridLayout()

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

	for (i, rf) in enumerate(reward_first_levels)	
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
let
	f = reliability_by_condition(
		PLT_data,
		x -> x.session == "1",
		x -> x.session == "2",
		"Test-retest",
		"Session 1",
		"Session 2"
	)

	save("results/single_p_QL_PMLE_test_retest_by_condition.png", f, pt_per_unit = 1)

	f
end

# ╔═╡ fff23ca1-35bc-4ff1-aea6-d9bb5ce86b1f
let
	f = reliability_by_condition(
		PLT_data,
		x -> (x.session == "1") & iseven(x.block),
		x -> (x.session == "1") & isodd(x.block),
		"Split-half",
		"Even blocks",
		"Odd block";
		supertitle = "Session 1 split-half reliability"
	)

	save("results/single_p_QL_PMLE_split_half_sess1_by_condition.png", f, pt_per_unit = 1)

	f

end

# ╔═╡ 8c9a0662-25af-4280-ad48-270458edb018
let
	f = reliability_by_condition(
		PLT_data,
		x -> (x.session == "2") & iseven(x.block),
		x -> (x.session == "2") & isodd(x.block),
		"Split-half",
		"Even blocks",
		"Odd block";
		supertitle = "Session 2 split-half reliability"
	)

	save("results/single_p_QL_PMLE_split_half_sess2_by_condition.png", f,
		pt_per_unit = 1)

	f
end

# ╔═╡ 479726b5-605b-494e-8ff2-4d569d0c0ddd
let 
	f = reliability_by_condition(
		shortened_PLT,
		x -> x.session == "1",
		x -> x.session == "2",
		"Test-retest",
		"Session 1",
		"Session 2";
		supertitle = "Early stopping simulated for all"
	)

	save("results/single_p_QL_PMLE_test_retest_early_stop_by_condition.png", f, 
		pt_per_unit = 1)

	f
end

# ╔═╡ 3a8f569d-0d8e-4020-9023-a123cad9d5de
let
	f = reliability_by_condition(
		shortened_PLT,
		x -> (x.session == "1") & iseven(x.block),
		x -> (x.session == "1") & isodd(x.block),
		"Split-half",
		"Even blocks",
		"Odd block";
		supertitle = "Session 1 split-half reliability\nearly stopping simulated for all"
	)

	save("results/single_p_QL_PMLE_split_half_sess1_early_stop_by_condition.png", f, 
		pt_per_unit = 1)

	f

end

# ╔═╡ 10565355-90ae-419c-9b92-8ff18fcd48b3
let
	f = reliability_by_condition(
		shortened_PLT,
		x -> (x.session == "2") & iseven(x.block),
		x -> (x.session == "2") & isodd(x.block),
		"Split-half",
		"Even blocks",
		"Odd block";
		supertitle = "Session 2 split-half reliability\nearly stopping simulated for all"
	)

	save("results/single_p_QL_PMLE_split_half_sess2_early_stop_by_condition.png", f, 
		pt_per_unit = 1)

	f
end

# ╔═╡ fa9f86cd-f9b1-43bb-a394-c105ba0a36fa
function reliability_by_valence(
	PLT_data::DataFrame
)	

	# Function to filter data
	filter1 = x -> x.session == "1"
	filter2 = x -> x.session == "2"

	filterr = x -> x.valence > 0
	filterp = x -> x.valence < 0

	# Compare test-retest between reward and punishment
	maps_reward = fit_split(filter(filterr, PLT_data), 
		filter1, filter2)

	test_retest = DataFrame(
		test_retest_reward_a = 
			bootstrap_correlation(maps_reward.a_1, maps_reward.a_2),
		test_retest_reward_ρ = 
			bootstrap_correlation(maps_reward.ρ_1, maps_reward.ρ_2)
	)

	maps_punishment = 
		fit_split(filter(filterp, PLT_data), filter1, filter2)

	test_retest.test_retest_punishment_a = 
		bootstrap_correlation(maps_punishment.a_1, maps_punishment.a_2)

	test_retest.test_retest_punishment_ρ = 
		bootstrap_correlation(maps_punishment.ρ_1, maps_punishment.ρ_2)

	# Compare reward and punishment within session
	maps_sess1 = fit_split(filter(filter1, PLT_data), 
		filterr, filterp)

	split_half = DataFrame(
		split_half_valences_asess1 = 
			bootstrap_correlation(maps_sess1.a_1, maps_sess1.a_2),
		split_half_valences_ρsess1= 
		bootstrap_correlation(maps_sess1.ρ_1, maps_sess1.ρ_2)
	)
	

	maps_sess2 = fit_split(filter(filter2, PLT_data), 
		filterr, filterp)

	split_half.split_half_valences_asess2 = 
		bootstrap_correlation(maps_sess2.a_1, maps_sess2.a_2)

	split_half.split_half_valences_ρsess2= 
		bootstrap_correlation(maps_sess2.ρ_1, maps_sess2.ρ_2)


	# Comapre odd and even blocks within session
	oddeven_sess1 = fit_split(filter(filter1, PLT_data), 
		x -> isodd(x.block), x -> iseven(x.block))

	split_half.split_half_oddeven_asess1= 
		bootstrap_correlation(oddeven_sess1.a_1, oddeven_sess1.a_2)

	split_half.split_half_oddeven_ρsess1= 
		bootstrap_correlation(oddeven_sess1.ρ_1, oddeven_sess1.ρ_2)

	
	oddeven_sess2 = fit_split(filter(filter2, PLT_data), 
		x -> isodd(x.block), x -> iseven(x.block))

	split_half.split_half_oddeven_asess2 = 
		bootstrap_correlation(oddeven_sess2.a_1, oddeven_sess2.a_2)

	split_half.split_half_oddeven_ρsess2= 
		bootstrap_correlation(oddeven_sess2.ρ_1, oddeven_sess2.ρ_2)


	# Reshape reliabilities for plot	

	function stack_cors(df::DataFrame; sort_cols::Bool = false)
		# Stack
		stacked = stack(
			df,
			:,
			variable_name = "cor_col"
		)

		if sort_cols
			sort!(stacked, :cor_col)
		end

		# Prepare variables for raincloud plot
		stacked.variable = (x -> split(x, "_")[4]).(stacked.cor_col)
		stacked.level = (x -> split(x, "_")[3]).(stacked.cor_col)
		stacked.level_id = indexin(stacked.level,
			unique(stacked.level)) |> (x -> map(y -> y::Int64, x))

		return stacked
	end

	test_retest = stack_cors(test_retest)

	# Plot test retest
	f = Figure(size = (700, 600))
	
	plot_cor_dist(
		f[3,1], 
		test_retest, 
		:value;
		ylabel = "Correlation",
		title = "Test-retest",
		colors = [:green, :red]
	)
	
	
	# Reshape and plot split half
	split_half = stack_cors(split_half; sort_cols = true)
	
	plot_cor_dist(
		f[3,2], 
		split_half, 
		:value;
		ylabel = "Correlation",
		title = "Split-half",
		xticks = repeat(["sess. 1", "sess. 2"], 2)
	)


	Label(
		f[4,2],
		"a",
		tellwidth = false,
		halign = 0.25
	)

	Label(
		f[4,2],
		"ρ",
		tellwidth = false,
		halign = 0.75
	)


	Legend(
		f[3,2],
		[PolyElement(color = c) for c in Makie.wong_colors()[1:2]],
		["odd / even", "valence"],
		"By",
		# orientation = :horizontal,
		halign = :right,
		valign = :bottom,
		framevisible = false,
		tellwidth = false,
		titleposition = :left
	)

	colsize!(f.layout, 2, Relative(2/3))
	
	# Plot parameter values
	function parameter_linked_valence_by_session!(
		f::GridPosition,
		maps_reward::DataFrame,
		maps_punishment::DataFrame,
		param::String
	)
		ax = Axis(
			f,
			ylabel = param,
			xticks = 1:2,
			xlabel = "Session"
		)
	
		function draw_scatter_segment!(
			ax::Axis,
			maps_reward::DataFrame,
			maps_punishment::DataFrame,
			x::Float64, # Position to plot on
			col::Symbol, # Column in DataFrames to plto
		)
	
			linesegments!(
				ax,
				repeat(x .+ [-.1, .1], nrow(maps_reward)),
				vcat([[maps_reward[i, col], maps_punishment[i, col]] 
					for i in 1:nrow(maps_reward)]...),
				color = (:grey, 0.3)
			)
		
			scatter!(
				ax,
				fill(x - 0.1, nrow(maps_reward)),
				maps_reward[!, col],
				color = :green,
				markersize = 6
			)
		
			scatter!(
				ax,
				fill(x + 0.1, nrow(maps_reward)),
				maps_punishment[!, col],
				color = :red,
				markersize = 6
			)

			boxplot!(
				ax,
				fill(x - .2, nrow(maps_reward)),
				maps_reward[!, col],
				color = :green,
				width = 0.1,
				show_outliers = false
			)

			boxplot!(
				ax,
				fill(x + .2, nrow(maps_reward)),
				maps_punishment[!, col],
				color = :red,
				width = 0.1,
				show_outliers = false
			)
		end
	
		draw_scatter_segment!(
			ax, 
			maps_reward, 
			maps_punishment, 
			1., 
			Symbol("$(param)_1")
		)
		draw_scatter_segment!(
			ax, 
			maps_reward, 
			maps_punishment, 
			2., 
			Symbol("$(param)_2")
		)
	end

	f_top = f[2,:] = GridLayout()
	parameter_linked_valence_by_session!(
		f_top[1,1],
		maps_reward,
		maps_punishment,
		"a"
	)

	parameter_linked_valence_by_session!(
		f_top[1,2],
		maps_reward,
		maps_punishment,
		"ρ"
	)

	Legend(
		f[0,:],
		[PolyElement(color = c) for c in [:green, :red]],
		["Reward", "Punishment"],
		orientation = :horizontal,
		framevisible = false,
		tellwidth = false,
		titleposition = :left
	)

	
	Label(
		f[1, :],
		"Parameter values",
		halign = :center,
		font = :bold
	)
	

	f

	
end

# ╔═╡ 80ae54ff-b9d2-4c30-8f62-4b7cac65201b
let
	f = reliability_by_valence(
		PLT_data
	)

	save("results/single_p_QL_PMLE_reliability_by_valence.png", f, 
		pt_per_unit = 1)

	f
end

# ╔═╡ 60866598-8564-4dd7-987c-fb74d3f3fc64
let
	f = reliability_by_valence(
		shortened_PLT
	)

	save("results/single_p_QL_PMLE_reliability_by_valence_shortened.png", f, 
		pt_per_unit = 1)

	f
end

# ╔═╡ 3b40c738-77cf-413f-9821-c641ebd0a13d
let
	f = reliability_by_valence(
		filter(x -> !x.valence_grouped, PLT_data)
	)

	save("results/single_p_QL_PMLE_reliability_by_valence_interleaved.png", f, 
		pt_per_unit = 1)

	f
end

# ╔═╡ a59e36b3-15b3-494c-a076-b3eade2cc315
function plot_q_learning_ppc_accuracy(
	data::DataFrame,
	ppc::DataFrame;
	title::String = ""
)

	f_acc = Figure()

	ax = Axis(f_acc[1,1],
        xlabel = "Trial #",
        ylabel = "Prop. optimal choice",
        xautolimitmargin = (0., 0.),
        title = title
    )

	# Plot bootstraps
	plot_group_accuracy!(ax, ppc;
		group = :bootstrap_idx,
		error_band = false,
		linewidth = 2.,
		colors = fill(:grey, length(unique(ppc.bootstrap_idx)))
	)

	# Plot data
	ax_acc = plot_group_accuracy!(ax, data;
		error_band = false
	)

	f_acc

end


# ╔═╡ 594b27ed-4bde-4dae-b4ef-9abc67bf699c
# Simulate multiple participants from (bootstrapped) posterior
function simulate_multiple_from_posterior_single_p_QL(
	bootstraps::DataFrame
)
	# Get task structures
	conds = unique(bootstraps.condition)

	conds = Dict(
		c => task_vars_for_condition(c) for c in conds
	)

	# Simulate for each bootstrap, each participant
	sim_data = []
	lk = ReentrantLock()

	Threads.@threads for i in 1:nrow(bootstraps)
		tsim = simulate_from_posterior_single_p_QL(
			bootstraps[i, :], 
			conds[bootstraps[i, :condition]],
			i
		)

		lock(lk) do
			push!(sim_data, tsim)
		end

	end


	sim_data = vcat(sim_data...)
	
end


# ╔═╡ 33fc2f8e-87d0-4e9e-9f99-8769600f3d25
let
	f = plot_q_learning_ppc_accuracy(
		PLT_data,
		simulate_multiple_from_posterior_single_p_QL(
			bootstrap_optimize_single_p_QL(
				PLT_data
			)
		)
	)

	save("results/single_p_QL_PMLE_bootstrap_PPC.png", f, 
		pt_per_unit = 1)
	f
end

# ╔═╡ db3cd8d3-5e46-48c6-b85d-f4d302fff690
f = plot_q_learning_ppc_accuracy(
		PLT_data,
		simulate_multiple_from_posterior_single_p_QL(
			bootstrap_optimize_single_p_QL(
				PLT_data;
				estimate = "MLE"
			)
		)
	)

# ╔═╡ 15bfde49-7b68-40fe-bebe-7a8b5c27e27e
let
	task = task_vars_for_condition("00")

	participants = DataFrame(
		a = rand(Normal(0., 0.1), 100), 
		ρ = rand(truncated(Normal(5., 0.5), lower = 0), 100), 
		bootstrap_idx = 1:100, 
		prolific_pid = 1:100,
		condition = fill("00", 100)
	)

	data = simulate_multiple_from_posterior_single_p_QL(
		participants
	) 

	data = leftjoin(data, participants[!, Not(:bootstrap_idx)], on = :prolific_pid)

	rename!(data,
		:optimal_right => :optimalRight,
		:feedback_left => :outcomeLeft,
		:feedback_right => :outcomeRight
	)


	bootstraps = bootstrap_optimize_single_p_QL(
				data;
				estimate = "MLE"
			)




	# plot_q_learning_ppc_accuracy(
	# 	data,
	# 	data	
	# )

end

# ╔═╡ Cell order:
# ╠═fb94ad20-57e0-11ef-2dae-b16d3d00e329
# ╠═261d0d08-10b9-4111-9fc8-bb84e6b4cef5
# ╠═fa79c576-d4c9-4c42-b5df-66d886e8abe4
# ╠═069177a1-d078-4d3c-bfa3-1bef04065d89
# ╟─f5113e3e-3bcf-4a92-9e76-d5eed8088320
# ╠═9477b295-ada5-46cf-b2e3-2c1303873081
# ╠═2afaba84-49b6-4770-a3bb-6e8e4c8be4ba
# ╠═50d88c5c-4a3f-4ded-81cf-600eddb3bbf9
# ╟─47b4f578-98ee-4ae0-8359-f4e8de5a63f1
# ╠═1d982252-c9ac-4925-9cd5-976456d32bc4
# ╠═2eb2dd61-abae-4328-9787-7a841d321836
# ╠═a3c8a90e-d820-4542-9043-e06a0ec9eaee
# ╠═c2e51103-b6ee-43a3-87e5-a9a2b28ed8e4
# ╠═48343b75-a0cd-4806-9867-b861b118491d
# ╠═8a567ab1-db8a-47df-ac2b-59714f230ae6
# ╠═0a151f69-f59e-48ae-8fc2-46a455e4f049
# ╠═fff23ca1-35bc-4ff1-aea6-d9bb5ce86b1f
# ╠═8c9a0662-25af-4280-ad48-270458edb018
# ╠═525522d1-5ced-46f2-8c9b-3299d3cb244d
# ╠═479726b5-605b-494e-8ff2-4d569d0c0ddd
# ╠═3a8f569d-0d8e-4020-9023-a123cad9d5de
# ╠═10565355-90ae-419c-9b92-8ff18fcd48b3
# ╠═a53db393-c9e7-4db3-a11d-3b244823d951
# ╠═a88ffe29-f0f4-4eb7-8fc3-a7fcc08560d0
# ╠═8ebaeb8e-e760-4f9e-a6bc-e3ecf44e6665
# ╠═80ae54ff-b9d2-4c30-8f62-4b7cac65201b
# ╠═60866598-8564-4dd7-987c-fb74d3f3fc64
# ╠═3b40c738-77cf-413f-9821-c641ebd0a13d
# ╠═5ebdc7b1-1c94-425d-9e08-0fc7da8cdc34
# ╠═c6558729-ed5b-440b-8e59-e69071b26f09
# ╠═33fc2f8e-87d0-4e9e-9f99-8769600f3d25
# ╠═db3cd8d3-5e46-48c6-b85d-f4d302fff690
# ╠═15bfde49-7b68-40fe-bebe-7a8b5c27e27e
# ╠═de41a8c1-fc09-4c33-b371-4d835a0a46ce
# ╠═2239dd1c-1975-46b4-b270-573efd454c04
# ╠═fa9f86cd-f9b1-43bb-a394-c105ba0a36fa
# ╠═a59e36b3-15b3-494c-a076-b3eade2cc315
# ╠═594b27ed-4bde-4dae-b4ef-9abc67bf699c
