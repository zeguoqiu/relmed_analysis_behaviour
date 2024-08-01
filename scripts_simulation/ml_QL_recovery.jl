### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 133edcd0-4ffc-11ef-3974-bd37beb0fb18
begin
	cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, StatsBase,
		StanSample, JSON, RCall, CSV, PlutoUI
	include("PLT_task_functions.jl")
	include("stan_functions.jl")
	include("plotting_functions.jl")
end

# ╔═╡ eb0db533-15ad-4252-8876-23913031a6d9
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

# ╔═╡ 1c6832a6-3504-4168-8da6-76b78a16bbbf
# Simulate participants with varying learning rates and reward sensitivities
sim_dat, true_params = let n_blocks = 24,
	n_trials = 13,
	feedback_magnitudes = [1., 0.5],
	feedback_ns = [7, 6],
	res = 30,
	a_vals = range(-2., 2., res),
	ρ_vals = range(0.1, 10., res)

	true_params = allcombinations(DataFrame, 
		:α => a2α.(a_vals), 
		:ρ => ρ_vals)
	
	sim_dat = simulate_q_learning_dataset(
		n_trials,
		repeat([feedback_magnitudes], n_blocks),
		repeat([feedback_ns], n_blocks),
		vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-2]],
			repeat([[n_trials-3]], n_blocks - 4)),
		true_params.α,
		true_params.ρ;
		random_seed = 0
	)

	sim_dat.isOptimal = (sim_dat.choice .== 1) .+ 0

	sim_dat, true_params
end

# ╔═╡ 3c94c732-df6f-4e3f-9f78-c6385bdbb8dd
# Fit the data
begin
	_, mle, mle_time = load_run_cmdstanr(
		"mle_recovery",
		"group_QLRs02ml.stan",
		to_standata(sim_dat,
			compute_avg_reward(sim_dat)[1];
			PID_col = :PID,
			outcome_col = :outcome
		);
		print_vars = vcat([["a[$i]", "rho[$i]"] for i in 1:5]...),
		threads_per_chain = 12,
		method = "optimize",
		iter_warmup = 500000
	)
	mle, mle_time

end

# ╔═╡ 99f36c35-aa16-4dfa-9b0f-9c5456808503
# Fit the data with penalized MLE
begin
	_, pmle, pmle_time = load_run_cmdstanr(
		"pmle_recovery",
		"group_QLRs02pml.stan",
		to_standata(sim_dat,
			compute_avg_reward(sim_dat)[1];
			PID_col = :PID,
			outcome_col = :outcome
		);
		print_vars = vcat([["a[$i]", "rho[$i]"] for i in 1:5]...),
		threads_per_chain = 12,
		method = "optimize",
		iter_warmup = 500000
	)
	pmle, pmle_time

end

# ╔═╡ 5b6f424a-2d53-438f-9b2c-625ad50c2773
# Plot recovery
function plot_mle_recovery(
	mle::DataFrame, # MLE estimates
	true_params::DataFrame;
	ms::Int64 = 4,
	mle_label = "MLE"
) 
	mle_params = extract_participant_params(mle; rescale = false)
	mle_params = DataFrame(
		:mle_α => a2α.(collect(mle_params["a"][1, :])),
		:mle_ρ => collect(mle_params["rho"][1, :])
	)

	pars = hcat(mle_params, true_params)

	f = Figure()

	# Bivariate MLE distribution
	ax_α = Axis(
		f[1,1],
		xlabel = "Learning rate $mle_label",
		ylabel = "Reward senstivity $mle_label"
	)

	scatter!(
		pars.mle_α,
		pars.mle_ρ,
		markersize = ms
	)

	# True param grid
	ax_α = Axis(
		f[1,2],
		xlabel = "True learning rate",
		ylabel = "True reward sensitivity"
	)

	scatter!(
		pars.α,
		pars.ρ,
		markersize = ms
	)

	# Compare MLE reward sensitivity to true reward sensitivity
	ax_ρ = Axis(
		f[2,2],
		xlabel = "True reward sensitivity",
		ylabel = "Reward sensitivity $mle_label"
	)

	scatter!(
		pars.ρ,
		pars.mle_ρ,
		markersize = ms
	)

	# Compare MLE learning rate to true learning rate
	ax_α = Axis(
		f[2,1],
		xlabel = "True learning rate",
		ylabel = "Learning rate $mle_label"
	)

	scatter!(
		pars.α,
		pars.mle_α,
		markersize = ms
	)

		# Compare MLE reward sensitivity to true reward sensitivity
	ax_ρ = Axis(
		f[2,2],
		xlabel = "True reward sensitivity",
		ylabel = "Reward sensitivity $mle_label"
	)

	scatter!(
		pars.ρ,
		pars.mle_ρ,
		markersize = ms
	)


	f

end

# ╔═╡ f7bb1d86-4244-4d0d-88d1-cc21ad106c08
let
	f = plot_mle_recovery(mle, true_params)
	save("results/MLE_recovery.png", 
		f, pt_per_unit = 1)
	f
end

# ╔═╡ ad6ad3cf-b2fa-46a2-970a-a232ce2de93f
let
	f = plot_mle_recovery(pmle, true_params; mle_label = "PMLE")
	save("results/PMLE_recovery.png", 
		f, pt_per_unit = 1)
	f
end

# ╔═╡ Cell order:
# ╠═133edcd0-4ffc-11ef-3974-bd37beb0fb18
# ╠═eb0db533-15ad-4252-8876-23913031a6d9
# ╠═1c6832a6-3504-4168-8da6-76b78a16bbbf
# ╠═3c94c732-df6f-4e3f-9f78-c6385bdbb8dd
# ╠═f7bb1d86-4244-4d0d-88d1-cc21ad106c08
# ╠═99f36c35-aa16-4dfa-9b0f-9c5456808503
# ╠═ad6ad3cf-b2fa-46a2-970a-a232ce2de93f
# ╠═5b6f424a-2d53-438f-9b2c-625ad50c2773
