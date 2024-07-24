### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ cf6fcac0-404f-11ef-1fa8-577bc61de858
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP
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

# ╔═╡ 5d546216-3888-48a8-9a89-0c97bdbc0cdc
# Engagement
begin
	f_FI = let n_blocks = 45, 
		n_trials = 13,
		feedback_magnitudes = [1., 2.],
		feedback_ns = [7, 6],
		res = 50,
		μ_a = range(-2., 1., res), 
		μ_ρ = range(0.01, 1.2, res),
		ns = 5
		
		FI = Q_learning_μ_σ_range_FI(n_blocks, n_trials, feedback_magnitudes, 
			feedback_ns, μ_a, μ_ρ)

		# Plot
		f_FI = Figure(size = (5.71, 2.85) .* 72)

		# Calculate alpha
		μ_α = a2α.(μ_a)
		
		ax_alpha = Axis(f_FI[1,1],
			xlabel = "Learning rate",
			ylabel = "Fisher Information"
		)

		lines!(ax_alpha,
			μ_α,
			FI[end, :],
			linewidth = 2
		)

		ax_rho = Axis(f_FI[1,2],
			xlabel = "Reward sensitivity",
			ylabel = "Fisher Information"
		)

		lines!(ax_rho,
			μ_ρ,
			FI[:, end],
			linewidth = 2
		)

		linkyaxes!(ax_alpha, ax_rho)

		f_FI
	end

	save("results/engagement_2lines.pdf", f_FI, pt_per_unit = 1)
	
	f_FI

end

# ╔═╡ Cell order:
# ╠═cf6fcac0-404f-11ef-1fa8-577bc61de858
# ╠═5d546216-3888-48a8-9a89-0c97bdbc0cdc
