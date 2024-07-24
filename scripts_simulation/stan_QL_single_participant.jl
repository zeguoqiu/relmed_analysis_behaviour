### A Pluto.jl notebook ###
# v0.19.42

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

# ╔═╡ bd5b4020-143d-11ef-28fb-f5108204e745
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, StatsBase,
		StanSample, JSON, RCall, CSV, PlutoUI
	include("task_functions.jl")
	include("stan_functions.jl")
	include("plotting_functions.jl")
end

# ╔═╡ 38dd7f41-8ac0-4d1f-831d-3112417b6d4c
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

# ╔═╡ cb86fa8d-39dd-473d-82bf-3cd9176b76b2
# Simulate one participant
begin
	n_blocks = 1000
	n_trials = 13
	feedback_magnitudes = [1., 2.]
	feedback_ns = [7, 6]
	a = 1.
	ρ = 0.8
	
	sim_dat = simulate_q_learning_dataset(1,
		n_trials,
		repeat([feedback_magnitudes], n_blocks),
		repeat([feedback_ns], n_blocks),
		vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-2]],
			repeat([[n_trials-3]], n_blocks - 4)),
		a,
		0.,
		ρ,
		0.
	)
	nothing

end

# ╔═╡ fe5dacf1-de3a-4ceb-9393-4b8045a3f76b
md"""
# Simulating and fitting single particpant Q learner

To begin, we simulate one dataset of $n_blocks blocks. We fit the whole dataset, as well as the first 10 or 100 blocks only.

As can be seen below, for the chosen values of α and ρ, recovery is good, with the true parameter values withing the 95% posterior interval (marked as a bold horizontal line beneath the posterior density plot). 

As expected, the poseterior for $n_blocks blocks is much more precise than for 100 blocks. The plot is cropped to the limits of the 100 block posterior. One can see that with 10 blocks, the posterior is very wide and shallow.
"""

# ╔═╡ 2bc42714-d182-467e-976e-ea426cb67e29
begin

	function fit_n_blocks(
		sim_dat::DataFrame,
		n::Int64
	)

		# Fit
		_, single_p_QL_draws= load_run_cmdstanr(
		"single_p_QL_$n",
		"single_p_QL.stan",
		to_standata(filter(x -> x.block <= n, sim_dat),
			feedback_magnitudes,
			feedback_ns);
		print_vars = ["a", "rho"]
		)

		single_p_QL_draws.α = cdf.(Normal(), single_p_QL_draws.a)

		return single_p_QL_draws
	
	end

	n_fits = [10, 100, 1000]
	fits = [fit_n_blocks(sim_dat, n) for n in n_fits]

	# Plot
	f_sim = Figure(size = (700, 300))

	for (i, p) in enumerate(["α", "rho"])

		ax = Axis(
			f_sim[1,i],
			xlabel = ["α", "ρ"][i],
			limits = ((minimum(fits[2][!, Symbol(p)]),
				maximum(fits[2][!, Symbol(p)])), nothing)
		)

		hideydecorations!(ax)
		hidespines!(ax, :l)

		for (j, d) in enumerate(fits)

			density!(ax,
				d[!, Symbol(p)],
				color = (:black, 0.),
				strokewidth = 2,
				strokecolor = Makie.wong_colors()[j]
			)

			linesegments!(ax,
				[(quantile(d[!, Symbol(p)], 0.025), 0.),
				(quantile(d[!, Symbol(p)], 0.975), 0.)],
				color = Makie.wong_colors()[j],
				linewidth = 4,
				label = string.(n_fits)[j]
		)
			
		end

		if i == 1
			axislegend("# of blocks",
				framevisible = false,
				position = (:left, :top)
			)
		end

		vlines!(ax,
			[cdf(Normal(), a), ρ][i],
			color = :gray,
			linestyle = :dash)

		Label(f_sim[0, 1:2],
			"95% PI and true value marked as dashed line",
			fontsize = 14
			)

		Label(f_sim[0, 1:2],
			"Posterior distributions\n\n\n",
			fontsize = 18
			)

		rowsize!(f_sim.layout, 0, Relative(0.15))

	end

	f_sim
	
end

# ╔═╡ 7677fef2-bc07-4087-8120-9ad15fbcfb07
begin
	n_sims = 60
	ns_blocks = [50, 100, 500]

	# Simulate, fit and summarise multiple times
	Random.seed!(0)
	sum_fits = DataFrame(
		vcat(
			[[simulate_fit_sum(i; n_blocks = n, name = "b$n") for i in 1:n_sims] for n in ns_blocks]...
		)
		
	)

end

# ╔═╡ ce0fac8d-b88d-4742-b7ff-4d1418f9bebf
md"""
## Parameter recovery by number of blocks

Now, we continue to simulate and fit Q learner across a range of parameter values.
We repeat this for three levels of block numbers.

Leftmost panel: We see that overall the estimated parameters correlate with the true parameter value. We do see a bit of regression to the mean for the learning rate (here plotted in unconstrained space as a).

Middle: Looking at the errors in parameter value estimation, the correlation between the error and the true value of a demostrates the regression to the mean effect.

Right: For both parameters, don't see a relationship between the posterior precicision of the parameter and the true value of the parameter, which is an encouraging sign. Overall, posteriors fit to 500 blocks are the most precise, as we would expect.


$(@bind show_n MultiCheckBox(ns_blocks, default = ns_blocks))
"""

# ╔═╡ 37b8e699-fa58-47cc-8abb-3a84c6df8704
begin
	plot_prior_predictive(sum_fits; show_n = show_n)
end

# ╔═╡ Cell order:
# ╠═bd5b4020-143d-11ef-28fb-f5108204e745
# ╟─38dd7f41-8ac0-4d1f-831d-3112417b6d4c
# ╠═cb86fa8d-39dd-473d-82bf-3cd9176b76b2
# ╟─fe5dacf1-de3a-4ceb-9393-4b8045a3f76b
# ╠═2bc42714-d182-467e-976e-ea426cb67e29
# ╠═7677fef2-bc07-4087-8120-9ad15fbcfb07
# ╟─ce0fac8d-b88d-4742-b7ff-4d1418f9bebf
# ╠═37b8e699-fa58-47cc-8abb-3a84c6df8704
