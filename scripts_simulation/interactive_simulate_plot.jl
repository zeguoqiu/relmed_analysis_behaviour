### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 7d46fa5c-0e09-11ef-0848-c92e44459854
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase
	include("AFC_task_functions.jl")
end

# ╔═╡ 8c5c8075-7423-45cd-b4f2-7005be0ed45a
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
	
	CairoMakie.activate!(type = "svg")
end


# ╔═╡ 6561ddcd-b076-4056-a7c0-d4bf28fb7d73
begin

	function string_to_vector(s)
		return eval(Meta.parse("[$s]"))
	end

	
	md"""
	# Task parameters
	Number of blocks
	$(@bind n_blocks Scrubbable(1:20, default = 8))

	Number of trials per block
	$(@bind n_trials Scrubbable(5:20, default = 13))
	"""
	
end

# ╔═╡ 6af24eef-5a8f-4412-a85c-98c689bd5fcc
begin
	default_feedback_ns = "$(floor(Int64, n_trials/2)), $(ceil(Int64, n_trials/2))"
	
	md"""
	Feedback magnitudes (in points)
	$(@bind feedback_magnitudes_s TextField(default = "1., 0.5"))

	Number of trials each magnitude appears
	$(@bind feedback_ns_s TextField(default = default_feedback_ns))
	
	Number of correct choices afterwhich to stop $(@bind stop_after Scrubbable(1:n_trials, default = n_trials))
	
	# Agent parameters
	## Learning rates
	Learning rates for each group are drawn according to the following scheme:
	```
		a ~ Normal(μ_a, σ_a)
		α = Φ(a)
	```

	| Set μ_a per group | Set σ_a per group |
	| -------- | ------- |
	| Group A $(@bind μ_a1 PlutoUI.Slider(-4:0.1:4, default = 0.3)) | Group A $(@bind σ_a1 PlutoUI.Slider(0.:0.01:0.5, default = 0.6)) |
	| Group B $(@bind μ_a2 PlutoUI.Slider(-4:0.1:4, default = 0.5)) | Group B $(@bind σ_a2 PlutoUI.Slider(0.:0.01:0.5, default = 0.6)) |
	| Group C $(@bind μ_a3 PlutoUI.Slider(-4:0.1:4, default = 0.2)) | Group C $(@bind σ_a3 PlutoUI.Slider(0.:0.01:0.5, default = 0.6)) |

	## Reward sensitivity
	Reward sensitivty for each group are drawn according to the following scheme:
	```
		ρ ~ Normal(μ_ρ, σ_ρ)
	```

	| Set μ_ρ per group | Set σ_ρ per group |
	| -------- | ------- |
	| Group A $(@bind μ_ρ1 PlutoUI.Slider(0.:0.01:6., default = 2.8)) | Group A $(@bind σ_ρ1 PlutoUI.Slider(0.:0.01:0.5, default = 2.)) |
	| Group B $(@bind μ_ρ2 PlutoUI.Slider(0.:0.01:6., default = 3.)) | Group B $(@bind σ_ρ2 PlutoUI.Slider(0.:0.01:0.5, default = 2.)) |
	| Group C $(@bind μ_ρ3 PlutoUI.Slider(0.:0.01:6., default = 2.6)) | Group C $(@bind σ_ρ3 PlutoUI.Slider(0.:0.01:0.5, default = 2.)) |
	"""
end

# ╔═╡ 18cf9667-9325-4291-809c-83f0456ca69d
begin
	# Parse input
	feedback_magnitudes = string_to_vector(feedback_magnitudes_s)
	feedback_ns = string_to_vector(feedback_ns_s)

	# Check input
	@assert length(feedback_magnitudes) == length(feedback_ns)
	@assert sum(feedback_ns) == n_trials

	# Simulate
	sim_dat = simulate_groups_q_learning_dataset(300,
		n_trials,
		[μ_a1, μ_a2, μ_a3],
		[σ_a1, σ_a2, σ_a3],
		[μ_ρ1, μ_ρ2, μ_ρ3],
		[σ_ρ1, σ_ρ2, σ_ρ3];
		feedback_magnitudes = repeat([feedback_magnitudes], n_blocks),
		feedback_ns = repeat([feedback_ns], n_blocks),
		feedback_common = n_blocks <= 4 ? 
			[[n_trials], 
				[n_trials-1], 
				[n_trials-1], 
				[n_trials-1]][1:n_blocks] :
			vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-1]], repeat([[n_trials-3]], n_blocks - 4)),

		stop_after = stop_after < n_trials ? stop_after : missing
	)	

	if stop_after < n_trials
		@info "$(round(sum(sim_dat.choice.== 0) / nrow(sim_dat) * 100, digits = 2))% of trials were skipped due to early stopping"
		filter!(x -> x.choice > 0, sim_dat)
	end

	# Plot
	f = Figure()


	plot_q_value_acc!(f, sim_dat)
end

# ╔═╡ 999d115a-e21f-41a5-8f43-685c9f216956
md"""
(Left) The evolution of Q values for each stimulus over trials, averaged over blocks. Thick lines denote group average, fine lines each participants' data. (Right) Proportion correct choice (stimulus A by convention). Lines denote group mean, bands span ±SEM over participants. 
"""

# ╔═╡ Cell order:
# ╠═7d46fa5c-0e09-11ef-0848-c92e44459854
# ╠═8c5c8075-7423-45cd-b4f2-7005be0ed45a
# ╟─6561ddcd-b076-4056-a7c0-d4bf28fb7d73
# ╟─6af24eef-5a8f-4412-a85c-98c689bd5fcc
# ╠═18cf9667-9325-4291-809c-83f0456ca69d
# ╟─999d115a-e21f-41a5-8f43-685c9f216956
