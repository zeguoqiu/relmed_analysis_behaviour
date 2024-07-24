### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ff13dfd4-1906-11ef-321b-59b69cfd64bc
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, Cbc, HiGHS
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations
	include("task_functions.jl")
	include("fisher_information_functions.jl")
	include("plotting_functions.jl")
end

# ╔═╡ 4df49829-b6d7-4757-ac19-1fd2d5a61ac2
md"""
# Estimating parameter recover of probabilistic learning task using simulations and Fisher information
"""

# ╔═╡ 6015a232-463c-4634-a7fa-1d6ff66071a4
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


# ╔═╡ d97480fc-5819-4f29-a7c5-c230b7f814e8
md"""
## Increasing the number of trials within a block vs. increasing number of blocks

### Single participant

Suspecting that the early trials are important in terms of conveying information about an agent, we simulated datasets with varying numbers of trials within a block, or varying numbers of blocks.

As can be clearly seen, increasing the number of blocks increases Fisher Information much more markedly than increasing the number of trials within a block by the same amount.

Additionally, we see that Fisher Information rises with the values of the learning rate α and reward sensitivity ρ. This undescores the importance of setting up the task such that participants will have high α and ρ values, that is, will be engaged, learning quickly and choosing actions in a principled manner.

Printed below is the summary of a simple linear regression model fit to Fisher information values. The `sum` and `diff` coefficients are for the sum and difference between α and ρ parameters respectively. The significant coefficient for the sum captures the dependence of Fisher information on the overall capability of the participant. The difference captures mostly scaling differeces between the parameters. `n_s` is the number of overall trials in the dataset, unsurprisingly, it is predictive of Fisher information. `factor_s` captures whether the trials are increased by adding blocks or by adding trials to each block. It's positive significant value indicates that adding blocks is prefereable. The interaction between `n_s` and `factor_s` indicates that the rise in Fisher information with added data is stepper when data is added as blocks as opposed to trials in a block.
"""

# ╔═╡ 80d317ac-05a2-4db5-8b21-36c2187a1c3a
begin
	let n_blocks = [10, 30, 60], 
		n_trials = [10, 30, 60],
		μ_a = range(-2., 1., 30), 
		μ_ρ = range(0.01, 1.2, 30),
		stop_after = missing

		# Simulate
		FI = hcat([Q_learning_μ_σ_range_FI(n, n_trials[2], [1., 2.], 
			[floor(Int64, n_trials[2]/2), ceil(Int64, n_trials[2]/2)], 
			μ_a, μ_ρ; stop_after = stop_after) for n in n_blocks],
		[Q_learning_μ_σ_range_FI(13, n, [1., 2.], 
			[floor(Int64, n/2), ceil(Int64, n/2)], 
			μ_a, μ_ρ; stop_after = stop_after) for n in n_trials])

		# Plot
		f_trials_single_p = Figure()

		plot_blocks_trials!(
			f_trials_single_p[1,1],
			FI,
			cdf.(Normal(), μ_a),
			μ_ρ;
			xlabel = ["α\n$(n)" for n in n_trials],
			ylabel = "ρ"
		)

		# Fit regression model
		nrows, ncols = size(FI[1,1])

		FI_df = vcat([DataFrame(
			μ_a = repeat(μ_a, inner = ncols),
			μ_ρ = repeat(μ_ρ, outer = nrows),
			FI = vec(FI[i,j]),
			n = n_blocks[i],
			factor = ["blocks", "trials"][j]
		) for i in 1:size(FI,1) for j in 1:size(FI,2)]...)

		# Treatment coding for factor
		FI_df.factor_s = (x -> x == "blocks" ? 1 : - 1).(FI_df.factor)

		# Standardize n for interaction
		FI_df.n_s = zscore(FI_df.n)

		# Transform α and ρ into sum and difference
		FI_df.sum = FI_df.μ_a + FI_df.μ_ρ
		FI_df.diff = FI_df.μ_a - FI_df.μ_ρ
		
		reg_mod = lm(@formula(FI ~ sum + diff + n_s * factor_s), FI_df)
		
		println(coeftable(reg_mod))
		
		f_trials_single_p

	end
	
end

# ╔═╡ c4aa18dd-6189-41c0-b143-7676a8551c49
md"""
### Group of participants

I repeated the same simulation with a group of 172 participants.

As can be seen below, Fisher information depends on the mean learning rate μ_α and mean reward sensitivity μ_ρ.

However, the standard deviations of these parameters in the group do not affect Fisher information. That is because the log likelihood is computed based on each participants' individual learning rate and reward sensitivity parameters, and is agnostic towards the hierarchical strucure of the data. Hence, I don't think Fisher information is a good measure of parameter recovery in this repsect. Accordingly, in following analyses I focus on the single participant level.
"""

# ╔═╡ 96720764-b861-473c-8221-032aba296d50
# ╠═╡ disabled = true
#=╠═╡
# Compute FI over grid of n_trials and n_blocks for group of participants
begin
	res = 6 # Resolution of plots
	group_ns = [10, 13, 16]
	group_μ_a = range(-2., 1., res) 
	group_μ_ρ = range(0.01, 1.2, res)
	group_σ_a = range(0.001, 0.3, res)
	group_σ_ρ = range(0.001, 0.3, res)
	
	group_FI = let n_trials = group_ns,
		n_blocks = group_ns,
		n_participants = 172,
		μ_a = group_μ_a,
		μ_ρ = group_μ_ρ,
		σ_a = group_σ_a,
		σ_ρ = group_σ_ρ
	
		hcat([Q_learning_μ_σ_range_FI(n, n_trials[2], [1., 2.], 
			[floor(Int64, n_trials[2]/2), ceil(Int64, n_trials[2]/2)], 
			μ_a, μ_ρ; 
			n_participants = n_participants,
			σ_a = σ_a,
		    σ_ρ = σ_ρ) for n in n_blocks],
		[Q_learning_μ_σ_range_FI(13, n, [1., 2.], 
			[floor(Int64, n/2), ceil(Int64, n/2)], 
			μ_a, μ_ρ; 
			n_participants = n_participants,
			σ_a = σ_a,
		    σ_ρ = σ_ρ) for n in n_trials])
		
	end

	nothing
end
  ╠═╡ =#

# ╔═╡ 64b5b79a-a335-40df-afb3-201cca4a3f36
# ╠═╡ disabled = true
#=╠═╡
begin
	FI_μs = avg_over.(group_FI, dims = (3,4))

	f_trials_μs = Figure()

	plot_blocks_trials!(
		f_trials_μs[1,1],
		FI_μs,
		cdf.(Normal(), group_μ_a),
		group_μ_ρ;
		xlabel = ["μ_α\n$(n)" for n in group_ns],
		ylabel = "μ_ρ"
	)

	f_trials_μs
end
  ╠═╡ =#

# ╔═╡ be3fd5a0-d6b0-479b-9232-d52ab65c71da
# ╠═╡ disabled = true
#=╠═╡
begin
	FI_a = avg_over.(group_FI, dims = (2,4))

	f_trials_as = Figure()

	plot_blocks_trials!(
		f_trials_as[1,1],
		FI_a,
		group_μ_a,
		group_σ_a;
		xlabel = ["μ_a\n$(n)" for n in group_ns],
		ylabel = "σ_a"
	)

	f_trials_as
end
  ╠═╡ =#

# ╔═╡ 46207f42-2481-419e-8885-019e539bbf07
# ╠═╡ disabled = true
#=╠═╡
begin
	FI_ρ = avg_over.(group_FI, dims = (1,3))

	f_trials_ρs = Figure()

	plot_blocks_trials!(
		f_trials_ρs[1,1],
		FI_ρ,
		group_μ_ρ,
		group_σ_ρ;
		xlabel = ["μ_ρ\n$(n)" for n in group_ns],
		ylabel = "σ_ρ"
	)

	f_trials_ρs
end
  ╠═╡ =#

# ╔═╡ 3cbc2f54-4566-44d5-b117-c9c78172585f
md"""
## Varying rewards

Next, we compare Fisher information when the number of reward values varies in the task.

Below, we simulate over a range of parameters with one level of reward (1.5), three levels (1, 1.5, 2), or five levels of reward (1, 1.25, 1.5, 1.75, 2). Crucially, the average reward in each of these is identical.

(Side note: for a given known α and ρ, higher reward results in higher Fisher information. The intuition here is that we higher absolute rewards, behaviour should be less stochastics, and so more diagnostic of underlying parameters. In most human studies, however, delivering larger rewards is not an option.)

The heatmaps below describe Fisher information across each of these three conditions. By eyeballing them, it seems that the values for 3 and 5 levels or reward are higher than for 1.

To better see this, I charted five regions along the diagonal of the Fisher information grid. These represent participants of rising levels of capabilities. Comparing 1, 3, and 5 levels of reward in each of these regions, we see that 3 and 5 are superior to 1. This effect however, is small.

Printed below is the output of simple linear regression model fit to Fisher information values. Again, sum and diff are the sum and difference of α and ρ values. n_rewards captures the effect of the number of different reward levels.
"""

# ╔═╡ 51d19690-41ca-442a-b490-015bc8d40ed5
begin
	
	function compute_reward_n(n::Number, k::Int)
	    q, r = divrem(k, n)  # quotient and remainder
	    v = [q + 1 for _ in 1:r]  # r elements of q+1
	    append!(v, [q for _ in 1:(n - r)])  # n-r elements of q
	    return round.(Int64, v)
	end
	
	
	let n_blocks = 45, 
		n_trials = 15,
		rewards = vcat([[1.5]], [collect(range(1., 2., n)) for n in [3, 5]]),
		μ_a = range(-2., 1., 30), 
		μ_ρ = range(0.01, 1.2, 30),
		ns = 5
		
		FI = [Q_learning_μ_σ_range_FI(n_blocks, n_trials, r, 
			compute_reward_n(length(r), n_trials), 
			μ_a, μ_ρ) for r in rewards]

		# Fit regression model
		nrows, ncols = size(FI[1,1])

		FI_df = vcat([DataFrame(
			μ_a = repeat(μ_a, inner = ncols),
			μ_ρ = repeat(μ_ρ, outer = nrows),
			FI = vec(FI[i]),
			n_rewards = [length(r) for r in rewards][i]
		) for i in 1:size(FI,1)]...)

		# Transform α and ρ into sum and difference
		FI_df.sum = FI_df.μ_a + FI_df.μ_ρ
		FI_df.diff = FI_df.μ_a - FI_df.μ_ρ
		
		reg_mod = lm(@formula(FI ~ sum + diff + n_rewards), FI_df)
		
		println(coeftable(reg_mod))

		# Plot
		f_rewards = Figure(size = (900, 580))

		# Plot FI heatmaps
		min_FI = minimum(map(mat -> minimum(mat), FI))
		max_FI = maximum(map(mat -> maximum(mat), FI))

		axs = []

		μ_α = cdf.(Normal(), μ_a)
		
		for (i, fi) in enumerate(FI)
			ax = Axis(f_rewards[1,i],
				xlabel = "α",
				ylabel = i == 1 ? "ρ" : "",
				title = "$(i == 1 ? "# of different rewards: " : "")$(length(rewards[i]))",
				titlealign = i == 1 ? :left : :center,
				aspect = 1.,
				xticks = (10:10:30, 
					["$t" for t in round.(μ_α[10:10:30], digits = 2)]),
				yticks = xticks = (10:10:30, 
					["$t" for t in round.(μ_ρ[10:10:30], digits = 2)])
			)

			push!(axs, ax)

			heatmap!(ax,
				fi,
				colorrange = (min_FI, max_FI)
			)

			# Plot squares
			n = 30 / ns
			for i in (0.5 + n/2):n:(30.5 - n/2)
				draw_square!(i, i, n, n; linewidth = 0.7)
			end
		end

		linkaxes!(axs...)

		# Plot summary
		
		sum_FI = [average_diagonal_squares(fi, round(Int64, 30/ns)) for fi in FI]

		gl = f_rewards[2,1:3] = GridLayout()


		for i in 1:ns
			sum_ax = Axis(gl[1,i],
				xlabel = "# of different rewards",
				ylabel = i == 1 ? "Fisher Information (nats)" : "",
				limits = ((0.62, 5.38), nothing),
				xticks = 1:2:5
			)

			
			med_FI = [s.med[i] for s in sum_FI]
			
			errorbars!(sum_ax,
				[1,3, 5],
				med_FI,
				med_FI - [s.lb[i] for s in sum_FI],
				[s.ub[i] for s in sum_FI] - med_FI)
	
			scatter!(sum_ax,
				[1,3, 5],
				med_FI)
		end

		rowsize!(f_rewards.layout, 1, Relative(0.6))
		
		f_rewards

	end
	
end

# ╔═╡ ab4f5a16-e2c5-4852-ac4a-b8fb0e75d9a8
begin
	let n_blocks = round.(Int64, range(sqrt(10), sqrt(500), 8).^2),
		n_trials = 13,
		ns = 5,
		res = 10,
		plot_ncol = 3,
		μ_a = range(-2., 1., res), 
		μ_ρ = range(0.01, 1.2, res),
		n_confusing = 3

		# Find all unique arrangements of n confusing trials within N trials
		feedback_commons_idx = combinations(1:n_trials, n_trials - n_confusing)

		feedback_commons = [[i in c ? 
			1 : 0 for i in 1:n_trials] for c in feedback_commons_idx]

		# Compute Fisher Information across parameter grid

		FI_file = "saved_models/FI/feedback_common_FI_diff_seed.jld2"
		if !isfile(FI_file)
			FI = [[Q_learning_μ_σ_range_FI(nb, n_trials, [1., 2.], 
				[floor(Int64, n_trials/2), ceil(Int64, n_trials/2)], 
				μ_a, μ_ρ; feedback_common = repeat([fc], nb),
				random_seeding = "different") for nb in n_blocks] for fc in feedback_commons]
			@save FI_file FI
		else
			@load FI_file FI
		end
		
		# Sammarize along diagonal
		n_in_diag = round(Int64, res / ns)
		sum_FI = [[average_diagonal_squares(fi, n_in_diag) for fi in ffi] for ffi in FI]

		# Flatten to get median of each region on FI matrix
		med_FI = [hcat([[s.med[i] for s in fsFI] for fsFI in sum_FI]...) for i in 1:ns]


		# Plot
		f_seq = Figure(size = (700, 550))

		row_n = 0.
		col_n = 0.

		for i in 1:ns

			# Compute Cronbach's alpha
			alpha = cronbachalpha(cov(transpose(med_FI[i]))).alpha

			# Compute average rank

			# Create Axis
			row_n = div(i - 1, plot_ncol) + 1
			col_n = rem(i - 1, plot_ncol) + 1
			
			sum_ax = Axis(f_seq[row_n, col_n],
				xlabel = "# blocks",
				ylabel = rem(i, plot_ncol) == 1 ? "Fisher information\nper block" : "",
				subtitle = "α ∈ [$(round(a2α(μ_a[i * n_in_diag - (n_in_diag - 1)]), 
					digits = 2)), $(round(a2α(μ_a[i * n_in_diag]), digits = 2))]\nρ ∈ [$(round(μ_ρ[i * n_in_diag - (n_in_diag - 1)], 
					digits = 2)), $(round(μ_ρ[i * n_in_diag], digits = 2))]\nCronbach's α=$(round(alpha, digits = 2))" 
			)

			# Normalize for color
			y_normalized = (med_FI[i] .- minimum(med_FI[i], dims=2)) ./ (maximum(med_FI[i], dims=2) .- minimum(med_FI[i], dims=2))

			# Plot medians
			for j in 1:size(med_FI[i], 2)
				lines!(sum_ax,
					n_blocks,
					med_FI[i][:, j] ./ n_blocks,
					color = y_normalized[:, j],
					colormap = :vik,
					linewidth = 0.5
				)
			end

		end

		# Compute Spearman's correlations across sections
		cor_avg_rank_med_FI = rotr90(cor(hcat([avg_over(hcat([tiedrank(row) for row in eachrow(matrix)]...), dims = (2,)) for matrix in med_FI]...)))

		ax_corr = Axis(f_seq[row_n, col_n + 1],
			yticks = (1:ns, string.(ns:-1:1)),
			subtitle = "Correlation of avg. ranks\nacross panels")
		hmap = heatmap!(ax_corr, cor_avg_rank_med_FI, colormap = :vik, 
			colorrange = (-1, 1))

		for i in 1:ns, j in 1:ns
    		txtcolor = cor_avg_rank_med_FI[i, j] > 0.5 ? :white : :black
		    text!(ax_corr, "$(round(cor_avg_rank_med_FI[i,j], digits = 2))", 
				position = (i, j),
		        color = txtcolor, align = (:center, :center), fontsize = 9)
		end

		f_seq

	
			
	end

end

# ╔═╡ 5ce0ac12-b290-4a23-b9e9-a83364359aab
vcat([missing], collect(3:2:12))

# ╔═╡ 015bf7f2-daf6-4fd7-852f-eacf8a27ed13
let n_blocks = round.(Int64, range(sqrt(10), sqrt(500), 8).^2),
	n_trials = 13,
	ns = 5,
	res = 10,
	plot_ncol = 3,
	μ_a = range(-2., 1., res), 
	μ_ρ = range(0.01, 1.2, res),
	stop_after = vcat(13, collect(3:2:10))

	# Compute Fisher Information across parameter grid

	FI = [[Q_learning_μ_σ_range_FI(nb, n_trials, [1., 2.], 
		[floor(Int64, n_trials/2), ceil(Int64, n_trials/2)], 
		μ_a, μ_ρ; random_seeding = "different", 
		stop_after = sa) for nb in n_blocks] for sa in stop_after]
	
	# Sammarize along diagonal
	n_in_diag = round(Int64, res / ns)
	sum_FI = [[average_diagonal_squares(fi, n_in_diag) for fi in ffi] for ffi in FI]

	# Flatten to get median of each region on FI matrix
	med_FI = [hcat([[s.med[i] for s in fsFI] for fsFI in sum_FI]...) for i in 1:ns]


	# Plot
	f_stop = Figure(size = (700, 550))

	row_n = 0.
	col_n = 0.

	sum_ax = nothing
	for i in 1:ns
		# Create Axis
		row_n = div(i - 1, plot_ncol) + 1
		col_n = rem(i - 1, plot_ncol) + 1
		
		sum_ax = Axis(f_stop[row_n, col_n],
			xlabel = "# blocks",
			ylabel = rem(i, plot_ncol) == 1 ? "Fisher information\nper block" : "",
			subtitle = "α ∈ [$(round(a2α(μ_a[i * n_in_diag - (n_in_diag - 1)]), 
				digits = 2)), $(round(a2α(μ_a[i * n_in_diag]), digits = 2))]\nρ ∈ [$(round(μ_ρ[i * n_in_diag - (n_in_diag - 1)], 
				digits = 2)), $(round(μ_ρ[i * n_in_diag], digits = 2))]" 
		)


		# Plot medians
		lns = []
		for j in 1:size(med_FI[i], 2)
			ln = lines!(sum_ax,
				n_blocks,
				med_FI[i][:, j] ./ n_blocks,
				color = stop_after[j],
				colormap = Reverse(:greens),
				colorrange = (minimum(stop_after), n_trials - 1),
				highclip = :magenta,
				label = stop_after[j] < n_trials ? string(stop_after[j]) : "never"
			)
			push!(lns, ln)
		end

	end

	lns = [LineElement(color = cgrad(:greens, length(stop_after)-1, rev = true)[i])
		for i in 1:(length(stop_after)-1)]
	lns = vcat([LineElement(color = :magenta)], lns)
	Legend(f_stop[row_n, col_n+1],
		lns,
		replace!(string.(stop_after), "$n_trials" => "never"),
		"Stop block after # correct",
		framevisible = false
	)


	f_stop
		
end

# ╔═╡ c4664db2-112b-4715-849e-afda4daa5fd7
# Understand what drives FI in sequence
let n_trials = 13,
	n_confusing = 3,
	stop_after = [missing, 5]

	# Prepare axes
	f_kernels = Figure(size = (29.32, 19.27) .* 72 ./ 2.54 ./ 1.3)

	axs = []
	for (i, (t, l)) in enumerate(zip([n_trials, n_trials-1], 
		["confusing trial\n", "autocorrelation"]))

		push!(axs, Axis(
			f_kernels[i,1],
			xlabel = "Trial #",
			ylabel = "Effect of $l on FI",
			xticks = 1:t
		))
	end

	# Compute and plot
	for (sa, ls, sa_label) in zip(stop_after, [:solid, :dash], ["never", "5 trials"])
	
		# Compute FI
		feedback_commons, avg_FI = compute_avg_zscored_FI(
			n_trials,
			n_confusing,
			stop_after = sa
		)
	
		# Transform sequnces into matrix
		feedback_commons = hcat(feedback_commons...) |> transpose |> collect
	
		# Code as 1 - confusing, 0 - common
		feedback_confusing = (1 .- feedback_commons)
	
		# Compute kernel over positions
		kernel = mean(feedback_confusing .* avg_FI, dims = 1)
	
		# Compute autocorrealtion of each sequnce
		autocor_feedback_confusing = mapslices(autocor, feedback_confusing, dims=2)
	
		# Compute kernel for autocorrelation
		autocor_kernel = mean(autocor_feedback_confusing .* avg_FI, dims = 1)
	
		# Plot
		for (i, (y, l)) in enumerate(zip([kernel, autocor_kernel], 
			["confusing trial\n", "autocorrelation"]))
			
			scatterlines!(axs[i], 
				1:length(y), 
				vec(y),
				color = vec(y),
				colormap = :redsblues,
				linestyle = ls,
				label = sa_label
			)
	
	
		end
	end

	Legend(f_kernels[0, 1],
		axs[1],
		"Stop after",
		tellwidth = false,
		orientation = :horizontal,
		framevisible = false,
		titleposition = :left
	)

	save("results/sequence_FI_kernels.pdf", f_kernels, pt_per_unit = 1)
	
	f_kernels

end

# ╔═╡ 8be48515-b7fc-4f14-a970-55c5640b56b0
let n_sessions = 1,
	n_blocks = 24,
	n_trials = 13,
	n_confusing = repeat(vcat([0 ,1, 1, 2], fill(3, n_blocks - 4)), n_sessions),
	categories = [('A':'Z')[div(i, 26) + 1] * ('a':'z')[rem(i, 26) + 1] for i in 1:(n_blocks + n_sessions)],
	ω_FI = 0.35,
	reward_magnitudes = [1., 2.],
	reward_ns = [floor(Int64, n_trials / 2), ceil(Int64, n_trials / 2)],
	stop_after = missing

	feedback_sequences = get_multiple_n_confusing_sequnces(
		n_trials,
		n_confusing,
		n_blocks_total,
		ω_FI,
		reward_magnitudes,
		reward_ns;
		stop_after = stop_after
	)

end

# ╔═╡ Cell order:
# ╠═4df49829-b6d7-4757-ac19-1fd2d5a61ac2
# ╠═ff13dfd4-1906-11ef-321b-59b69cfd64bc
# ╠═6015a232-463c-4634-a7fa-1d6ff66071a4
# ╟─d97480fc-5819-4f29-a7c5-c230b7f814e8
# ╠═80d317ac-05a2-4db5-8b21-36c2187a1c3a
# ╠═c4aa18dd-6189-41c0-b143-7676a8551c49
# ╠═96720764-b861-473c-8221-032aba296d50
# ╠═64b5b79a-a335-40df-afb3-201cca4a3f36
# ╠═be3fd5a0-d6b0-479b-9232-d52ab65c71da
# ╠═46207f42-2481-419e-8885-019e539bbf07
# ╠═3cbc2f54-4566-44d5-b117-c9c78172585f
# ╠═51d19690-41ca-442a-b490-015bc8d40ed5
# ╠═ab4f5a16-e2c5-4852-ac4a-b8fb0e75d9a8
# ╠═5ce0ac12-b290-4a23-b9e9-a83364359aab
# ╠═015bf7f2-daf6-4fd7-852f-eacf8a27ed13
# ╠═c4664db2-112b-4715-849e-afda4daa5fd7
# ╠═8be48515-b7fc-4f14-a970-55c5640b56b0
