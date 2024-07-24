### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 8cde3104-4362-11ef-0acc-4dab68887cdc
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations
	include("task_functions.jl")
	include("fisher_information_functions.jl")
	include("plotting_functions.jl")


	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")

	th = Theme(
		font = "Helvetica",
		fontsize = 28,
    	Axis = (
			xgridvisible = false,
			ygridvisible = false,
			rightspinevisible = false,
			topspinevisible = false,
			xticklabelsize = 24,
			yticklabelsize = 24,
			spinewidth = 1.5,
			xtickwidth = 1.5,
			ytickwidth = 1.5
    	)
	)
	set_theme!(th)

end

# ╔═╡ 96dbb563-cec1-40a1-9948-1ff325ee5a54
# Plot Q learners
f_q_learner = let feedback_magnitudes = [1., 2.],
		feedback_ns = [7, 6],
		n_trials = 13,
		n_blocks = 10,
		σ_a = 0.6,
		σ_ρ = 2.,
		μ_a = -.2
		μ_ρ = 2.2

		# Simulate dataset
		sim_dat = simulate_groups_q_learning_dataset(140,
			n_trials,
			repeat([feedback_magnitudes], n_blocks),
			repeat([feedback_ns], n_blocks),
				vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-1]], fill([n_trials-3], n_blocks - 4)),
			[μ_a],
			[σ_a],
			[μ_ρ],
			[σ_ρ]
		)
	
		# Plot
		f_q_learner = Figure(size = (386, 161) .* 72 ./ 25.4)
		plot_q_value_acc!(f_q_learner, 
			sim_dat;
			legend = false,
			colors = ["#34C6C6"]
		)

	 save("results/BAP_q_learners.pdf", f_q_learner, pt_per_unit = 1)

	f_q_learner
end

# ╔═╡ e045f929-54fa-4fb6-b59d-074cae245175
# Engagement
begin
	f_FI = let n_blocks = 24, 
		n_trials = 13,
		feedback_magnitudes = [1., 2.],
		feedback_ns = [7, 6],
		res = 75,
		μ_a = range(-2., 1., res), 
		μ_ρ = range(0.01, 3., res)
		
		FI = Q_learning_μ_σ_range_FI(n_blocks, n_trials, feedback_magnitudes, 
			feedback_ns, μ_a, μ_ρ)

		# Plot
		f_FI = Figure(size = (231, 127.5) .* 72 ./ 25.4)

		# Calculate alpha
		μ_α = a2α.(μ_a)
		
		ax_alpha = Axis(f_FI[1,1],
			xlabel = "Learning rate",
			ylabel = "Fisher Information"
		)

		find_closest_index(arr, x) = argmin(abs.(arr .- x))

		lines!(ax_alpha,
			μ_α,
			FI[find_closest_index(μ_ρ, 2.84), :],
			linewidth = 3,
			color = "#34C6C6"
		)

		ax_rho = Axis(f_FI[1,2],
			xlabel = "Reward sensitivity",
			ylabel = ""
		)

		lines!(ax_rho,
			μ_ρ,
			FI[:, find_closest_index(μ_a, 0.34)],
			linewidth = 3,
			color = "#34C6C6"
		)

		linkyaxes!(ax_alpha, ax_rho)

		f_FI
	end

	save("results/BAP_engagement_2lines.pdf", f_FI, pt_per_unit = 1)
	
	f_FI

end

# ╔═╡ 9438ba78-8aa1-4299-9389-c73cd434a01b


# ╔═╡ Cell order:
# ╠═8cde3104-4362-11ef-0acc-4dab68887cdc
# ╠═96dbb563-cec1-40a1-9948-1ff325ee5a54
# ╠═e045f929-54fa-4fb6-b59d-074cae245175
# ╠═9438ba78-8aa1-4299-9389-c73cd434a01b
