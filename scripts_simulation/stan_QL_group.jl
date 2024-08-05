### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 285b7932-112e-11ef-13c2-fba601473764
begin
	cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, StatsBase,
		StanSample, JSON, RCall, CSV
	include("PLT_task_functions.jl")
	include("stan_functions.jl")
	include("plotting_functions.jl")
end

# ╔═╡ 6dc9d218-1647-4802-9842-7c815cb44afb
# Set theme and general Makie settings
begin
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")

	th = Theme(
		font = "Helvetica",
    	Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false
    	)
	)
	set_theme!(th)
	
end

# ╔═╡ f3c846dc-698c-40b1-a60d-ef66eefb6d24
md"""
# Simulating and fitting multiple Q learners

Here, we test model fitting for hierarchical Q learners.

## Simple model fit and parameter recovery 

First, simulate a single dataset with 172 participants, and check for hyper-parameter recovery.
"""

# ╔═╡ 1975aff3-11a5-4b69-9b73-f61425bfd531
# Simulate one data set
begin
	n_blocks = 100
	n_trials = 13
	sequence_file = "results/PLT_task_structure_00.csv"
	μ_a = -0.6
	σ_a = 0.3
	μ_ρ = 6.5
	σ_ρ = 3.

	# Get task from sequence used for pilot 1
	task = DataFrame(CSV.File(sequence_file))

	# Replicate task 10 times
	function task_replicate(task::DataFrame, i::Int64)
		ntask = select(task, [:session, :block, :trial, :feedback_A, :feedback_B, :optimal_A])

		ntask.block = ntask.block .+ (i - 1) * maximum(ntask.block)

		return ntask

	end

	task = vcat([task_replicate(task, i) for i in 1:5]...)

	# Simulate 120 blocks
	sim_dat_large = simulate_q_learning_dataset(172,
		task,
		μ_a,
		σ_a,
		μ_ρ,
		σ_ρ
	)

	sim_dat_large.isOptimal = (sim_dat_large.choice .== 1) .+ 0
	

	# And subsample to first 24
	sim_dat = filter(x -> x.block <= 24, sim_dat_large);
end

# ╔═╡ 2b68c173-2879-42a0-a8a6-5152faafdf2b
begin
	group_QL_sum, group_QL_draws = load_run_cmdstanr(
		"group_QL",
		"group_QL.stan",
		to_standata(sim_dat,
			x -> repeat([sum(feedback_magnitudes .* feedback_ns) / 
			(sum(feedback_ns) * 2)], 2);
			PID_col = :PID,
			outcome_col = :outcome
		);
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho",
			"a[1]", "a[2]", "a[3]", "r[1]", "r[2]", "r[3]"]
	)
	group_QL_sum
end

# ╔═╡ ae92d50f-ae2a-4d91-b6d0-36da071bcde4
begin
	hyperparam_labels = [rich("μ", subscript("a")), rich("σ", subscript("a")), 
		rich("μ", subscript("ρ")), rich("σ", subscript("ρ"))]
	plot_posteriors(
		[group_QL_draws],
		["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		labels = hyperparam_labels,
		true_values = [μ_a, σ_a, μ_ρ, σ_ρ]
	)
end

# ╔═╡ 5e8d9995-b7b1-4fb7-953c-09e0c62bd05d
md"""
## Parallelized model
Next, we test a parallelized model, that runs faster.

We repeat the simulation and fit procedure, this time also fitting a dataset where each participant completes 100 blocks, for comparison.

Again we see good recovery of hyper parameters, and see that unsurprisingly, more data results in narrower posteriors, especially for the learning rate.
"""

# ╔═╡ 0b06c81e-7df2-40da-b335-622a4d696153
begin
	group_QLrs_sum, group_QLrs_draws, group_QLrs_time = load_run_cmdstanr(
		"group_QLrs",
		"group_QLrs.stan",
		to_standata(sim_dat,
			x -> repeat([sum(feedback_magnitudes .* feedback_ns) / 
			(sum(feedback_ns) * 2)], 2);
			PID_col = :PID,
			outcome_col = :outcome,
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	group_QLrs_sum, group_QLrs_time
end

# ╔═╡ 8d6146f2-48c1-4683-bfd2-bba2f077f7c0
begin	
	group_QLrs100_sum, group_QLrs100_draws, group_QLrs100_time = load_run_cmdstanr(
		"group_QLrs100",
		"group_QLrs.stan",
		to_standata(sim_dat_100,
			x -> repeat([sum(feedback_magnitudes .* feedback_ns) / 
			(sum(feedback_ns) * 2)], 2);
			PID_col = :PID,
			outcome_col = :outcome,
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	nothing
end

# ╔═╡ d279e83b-85c3-4dba-aab0-aa8e0dfe94ce
plot_posteriors(
	[group_QLrs_draws, group_QLrs100_draws],
	["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
	true_values = [μ_a, σ_a, μ_ρ, σ_ρ],
	labels = hyperparam_labels,
	model_labels = ["10 block", "100 blocks"]
)

# ╔═╡ 14088c9e-57f4-483e-a743-7e15461f4731
md"""
Next, looking at recovery for single-participant parameters. Below I plot parameters for eight participants.

The model parameterizes these single-participant parameters on a standardized scale, and then multiplies them by the population standard deviation and mean to get unstandardized values for RL. This is good practice for MCMC sampling.

The first plot shows reward sensitivity values on the standardized scale, the second plot shows them on the unstandardized scale. No interesting difference here, we see that the posteriors are quite wide with 10 blocks, and we get more distinction between participants with 100 blocks.
"""

# ╔═╡ b7d33b38-3c85-41c4-9902-c0d377b06e51
plot_posteriors(
	[group_QLrs_draws, group_QLrs100_draws],
	["rho[$n]" for n in 1:8],
	nrows = 2,
	labels = ["r$n" for n in 1:8],
	model_labels = ["10 block", "100 blocks"]
)

# ╔═╡ 59e8fda6-f3a0-4ab3-ac38-c7b8afa5ed5c
plot_posteriors(
	[group_QLrs_draws, group_QLrs100_draws],
	["rho[$n]" for n in 1:8],
	nrows = 2,
	labels = ["ρ$n" for n in 1:8],
	scale_col = :sigma_rho,
	mean_col = :mu_rho,
	model_labels = ["10 block", "100 blocks"]
)

# ╔═╡ f7564178-f850-4063-9b9a-075ff82497fd
md"""
Now, looking at the same for learning rate.
Looking first at unstandardized single-participant learning rates (on the inverse Phi scale) we see that the posteriors are quite wide, and only slightly narrower for the 100 block dataset.
"""

# ╔═╡ 36c261f1-f2e3-4e36-a409-bb57cbb71c85
plot_posteriors(
	[group_QLrs_draws, group_QLrs100_draws],
	["a[$n]" for n in 1:8],
	nrows = 2,
	labels = ["a$n" for n in 1:8],
	model_labels = ["10 block", "100 blocks"]
)

# ╔═╡ 95954638-0306-4fb3-8f45-c0c087151f09
md"""
On the unstandardized scale, there is hardly a difference between 10 and 100 blocks. Even sometimes an advantage to 10 blocks! It seems that we have unwarrented certainty in the value of individiaul learning rates. Note that they are all centered on the same value more or less - the group mean.
"""

# ╔═╡ 12ba8eba-5a1b-43a7-ab8b-675781542865
plot_posteriors(
	[group_QLrs_draws, group_QLrs100_draws],
	["a[$n]" for n in 1:8],
	nrows = 2,
	labels = ["a$n" for n in 1:8],
	scale_col = :sigma_a,
	mean_col = :mu_a,
	model_labels = ["10 block", "100 blocks"]
)

# ╔═╡ 13c208f9-9ca7-46f0-a8d8-85ae5eae3838
md"""
Now plotting this differently - looking at the entire dataset.
We can see that for 10 blocks, it is hard to ditinguish one participant from the next (left panel), and that the correlation between the true learning rate value and the estimated one is low.

Things somewhat improve for 100 blocks.
"""

# ╔═╡ 02b9a2a0-46f9-46cb-8bd4-40aa982ba8c5
begin
	sim_dat.a = quantile(Normal(), sim_dat.α)
	sim_dat.rho = sim_dat.ρ


	function plot_p_est_10_100(param::String)
		f_p = Figure()
	
		fp_axes = []
		for (i, d) in enumerate([group_QLrs_draws, group_QLrs100_draws])
			axs = plot_p_params!(
				f_p[i,1],
				d,
				sim_dat,
				param;
				ylabel = rich(rich("$([10, 100][i]) blocks", font = :bold), "\nParticipant #")
			)
			push!(fp_axes, axs)
		end
	
		linkaxes!([a[1] for a in fp_axes]...)
	
		return f_p
	end

	plot_p_est_10_100("a")
end

# ╔═╡ cb08c28e-9074-4913-a231-2ba3bbc10670
md"""
For reward sensitivity, things look better. We can distinguish between participants already with 10 blocks, and the correlation with true values is quite good.
"""

# ╔═╡ ed185b55-e824-4187-89fa-a3da1350f8c0
plot_p_est_10_100("rho")

# ╔═╡ e07604ed-945d-469c-b8ea-850781929c90
begin
	n_sims = 100

	# Simulate, fit and summarise multiple times
	Random.seed!(0)
	sum_fits = DataFrame([simulate_fit_sum(i; 
		model = "group_QLrs",
		n_participants = 172,
		prior_σ_a = Uniform(0.001, 0.8),
		prior_σ_ρ = Uniform(0.001, 0.8),
		) for i in 1:n_sims],
		
	)
	nothing

end

# ╔═╡ 0e6cf135-8871-4edf-b495-0c4395eb7f88
md"""
## Repeating simulation and fit across the parameter range
Below, we see the results from simulating and fitting data 100 times, across a grid of parameter values.

We see that recovery for μ_a and μ_ρ is generally good, with estimated values correlating very well with true values. There is slight overestimation of mu_a when the true value is very low.

We see that σ_a is not recovered well at all. This is the root of the problems we saw above the individual a estimates, and is refleceted in a_1 below as well.

σ_ρ is recovered well.
"""

# ╔═╡ fb13db5f-de92-4f99-ba7b-e3d789299482
plot_prior_predictive(sum_fits;
	params = ["mu_a", "mu_rho", "sigma_a", "sigma_rho", "a[1]", "rho[1]"],
	labels = [rich("μ", subscript("a")), rich("μ", subscript("ρ")),
		rich("σ", subscript("a")), rich("σ", subscript("ρ")),
		rich("a", subscript("1")), rich("ρ", subscript("1"))]
)

# ╔═╡ Cell order:
# ╠═285b7932-112e-11ef-13c2-fba601473764
# ╠═6dc9d218-1647-4802-9842-7c815cb44afb
# ╟─f3c846dc-698c-40b1-a60d-ef66eefb6d24
# ╠═1975aff3-11a5-4b69-9b73-f61425bfd531
# ╠═2b68c173-2879-42a0-a8a6-5152faafdf2b
# ╠═ae92d50f-ae2a-4d91-b6d0-36da071bcde4
# ╟─5e8d9995-b7b1-4fb7-953c-09e0c62bd05d
# ╠═0b06c81e-7df2-40da-b335-622a4d696153
# ╠═8d6146f2-48c1-4683-bfd2-bba2f077f7c0
# ╠═d279e83b-85c3-4dba-aab0-aa8e0dfe94ce
# ╟─14088c9e-57f4-483e-a743-7e15461f4731
# ╠═b7d33b38-3c85-41c4-9902-c0d377b06e51
# ╠═59e8fda6-f3a0-4ab3-ac38-c7b8afa5ed5c
# ╟─f7564178-f850-4063-9b9a-075ff82497fd
# ╠═36c261f1-f2e3-4e36-a409-bb57cbb71c85
# ╟─95954638-0306-4fb3-8f45-c0c087151f09
# ╠═12ba8eba-5a1b-43a7-ab8b-675781542865
# ╟─13c208f9-9ca7-46f0-a8d8-85ae5eae3838
# ╠═02b9a2a0-46f9-46cb-8bd4-40aa982ba8c5
# ╟─cb08c28e-9074-4913-a231-2ba3bbc10670
# ╠═ed185b55-e824-4187-89fa-a3da1350f8c0
# ╟─e07604ed-945d-469c-b8ea-850781929c90
# ╟─0e6cf135-8871-4edf-b495-0c4395eb7f88
# ╠═fb13db5f-de92-4f99-ba7b-e3d789299482
