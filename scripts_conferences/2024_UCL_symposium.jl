### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 5d9f27b4-2caf-11ef-3bee-51c614491e13
begin
	cd("/home/jovyan/")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations
	include("PLT_task_functions.jl")
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

# ╔═╡ 46595874-4cd8-4500-846f-93afbf3d264f
# Illustrate shaping protocol
begin
	conf = vcat([0, 1, 1, 2], fill(3, 6))
	common = 13 .- conf

	f_shaping = Figure(size = (204, 82) .* 72 ./ 25.4,
		figure_padding = (0,0,0,4))

	ax = Axis(f_shaping[2,1],
		xlabel = "Block #",
		ylabel = "# Trials",
		yautolimitmargin = (0., 0.),
		yticks = round.(range(0,13, 4))
	)

	colors = ["#34C6C6", "#FFCA36"]

	barplot!(ax,
		repeat(1:10, 2),
		vcat(common, conf),
		stack = vcat(fill(1, 10), fill(2, 10)),
		color = colors[vcat(fill(1, 10), fill(2, 10))]
	)

	Legend(f_shaping[1,1],
			[[PolyElement(color = colors[1])],
			[PolyElement(color = colors[2])]],
		[["Common"],
		["Confusing"]],
		"Feedback:",
		tellwidth = false,
		framevisible = false,
		orientation = :horizontal,
		titleposition = :left,
		titlefont = :regular,
		labelsize = 24,
		titlesize = 24)

	rowsize!(f_shaping.layout, 1, Relative(0.1))

	save("results/shaping_design.pdf", f_shaping, pt_per_unit = 1)

	f_shaping

end

# ╔═╡ 9022f0c9-16a4-40eb-81e8-eed9f2c03cc4
# Plot Q learners
begin
	f_q_learner = let feedback_magnitudes = [1., 2.],
		feedback_ns = [7, 6],
		n_trials = 13,
		n_blocks = 10,
		σ = 0.02,
		μ_a1 = -.2, μ_a2 = 1., μ_a3 = -.2,
		μ_ρ1 = 1.2, μ_ρ2 = 1.2, μ_ρ3 = 0.8

		# Compute alpha values to report 
		println(a2α.([μ_a1, μ_a2, μ_a3]))

		# Simulate dataset
		sim_dat = simulate_groups_q_learning_dataset(172,
			n_trials,
			[μ_a1, μ_a2, μ_a3],
			fill(σ, 3),
			[μ_ρ1, μ_ρ2, μ_ρ3],
			fill(σ, 3);
			feedback_magnitudes = repeat([feedback_magnitudes], n_blocks),
			feedback_ns = repeat([feedback_ns], n_blocks),
			feedback_common = vcat([[n_trials], [n_trials-1], [n_trials-1], [n_trials-1]], fill([n_trials-3], n_blocks - 4)),

		)
	
		# Plot
		f = Figure(size = (386, 161) .* 72 ./ 25.4)
		plot_sim_q_value_acc!(f, 
			sim_dat;
			legend = false,
			colors = ["#000000", "#52C152", "#34C6C6"]
		)
	end

	save("results/q_learners.pdf", f_q_learner, pt_per_unit = 1)

	f_q_learner
end

# ╔═╡ 98b42ee5-f343-4b3f-af52-a2e3fff53409
# Plot single FI matrix
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
		f_FI = Figure(size = (175, 175) .* 72 ./ 25.4)

		# Calculate alpha
		μ_α = a2α.(μ_a)
		
		ax = Axis(f_FI[1,1],
			xlabel = "Learning rate α",
			ylabel = "Reward sensitivity ρ",
			aspect = 1.,
			xticks = (10:10:(res-1), 
				["$t" for t in round.(μ_α[10:10:(res-1)], digits = 2)]),
			yticks = xticks = (10:10:(res-1), 
				["$t" for t in round.(μ_ρ[10:10:(res-1)], digits = 2)])
		)

		heatmap!(ax,
			FI
		)

		# Plot squares
		n = res / ns
		for i in (0.5 + n/2):n:(res + .5 - n/2)
			draw_square!(i, i, n, n; linewidth = 2, color = :white)
		end

		f_FI
	end

	save("results/FI.pdf", f_FI, pt_per_unit = 1)
	
	f_FI

end

# ╔═╡ 273f93e0-208f-4dd1-ba7d-2d14bc5d2090
begin
	# Plot blocks and trials
	let n_range = round.(Int64, range(15, 75, 5)), 
		res = 30,
		μ_a = range(-2., 1., res), 
		μ_ρ = range(0.01, 1.2, res),
		ns = 5,
		colors = ["#52C152", "#34C6C6"],
		linestyles = [:solid, :dash]

		# Simulate - first row blocks, second trials
		FI = hcat([Q_learning_μ_σ_range_FI(n, n_range[1], [1., 2.], 
			[floor(Int64, n_range[1]/2), ceil(Int64, n_range[1]/2)], 
			μ_a, μ_ρ) for n in n_range],
		[Q_learning_μ_σ_range_FI(n_range[1], n, [1., 2.], 
			[floor(Int64, n/2), ceil(Int64, n/2)], 
			μ_a, μ_ρ) for n in n_range])

		# Plot summary
		f_trials_blocks = Figure(size = (386, 110) .* 72 ./ 25.4)
		
		sum_FI = [average_diagonal_squares(fi, round(Int64, res/ns)) for fi in FI]

		med_FI = [s.med[4] for s in sum_FI]

		n_blocks = [n_range; fill(n_range, length(n_range))]

		for i in 1:ns
			sum_ax = Axis(f_trials_blocks[1,i],
				ylabel = i == 1 ? "Fisher information\n per block" : ""
			)
			
			med_FI = [s.med[i] for s in sum_FI]
			lb_FI = [s.lb[i] for s in sum_FI]
			ub_FI = [s.ub[i] for s in sum_FI]

			for j in 1:size(med_FI, 2)
			
				rangebars!(sum_ax,
					n_range,
					lb_FI[:, j] ./ n_blocks[j, :],
					ub_FI[:, j] ./ n_blocks[j, :],
					linewidth = 3,
					color = colors[j]
				)
		
				lines!(sum_ax,
					n_range,
					med_FI[:,j] ./ n_blocks[j, :],
					linestyle = linestyles[j],
					linewidth = 4,
					color = colors[j]
					)
			end
		end

		# Add legend on top
		Legend(f_trials_blocks[0,1:ns],
			[LineElement(color = c, 
				linewidth = 4,
				linestyle = s) for (c, s) in zip(colors, linestyles)],
			["blocks", "trials"],
			"Adding",
			orientation = :horizontal,
			titleposition = :left,
			framevisible = false,
			titlefont = :regular,
			labelsize = 24,
			titlesize = 24
		)

		save("results/trials_blocks.pdf", f_trials_blocks, pt_per_unit = 1)
		f_trials_blocks

	end


end

# ╔═╡ 7554ea02-c54b-4fb1-9d07-79c49c891a80
# Plot sequences
let n_blocks = round.(Int64, range(sqrt(10), sqrt(500), 8).^2),
	n_trials = 13,
	ns = 5,
	res = 10,
	plot_ncol = 3,
	μ_a = range(-2., 1., res), 
	μ_ρ = range(0.01, 1.2, res),
	n_confusing = 3,
	UCL_gradient = cgrad(["#002248", "#34C6C6", "#B6DCE5", 
					"#DEB8C3", "#AC145A", "#4B0A42"])

	# Find all unique arrangements of n confusing trials within N trials
	feedback_commons_idx = combinations(1:n_trials, n_trials - n_confusing)

	feedback_commons = [[i in c ? 
		1 : 0 for i in 1:n_trials] for c in feedback_commons_idx]

	# Compute Fisher Information across parameter grid

	FI_file = "saved_models/FI/feedback_common_FI.jld2"
	if !isfile(FI_file)
		FI = [[Q_learning_μ_σ_range_FI(nb, n_trials, [1., 2.], 
			[floor(Int64, n_trials/2), ceil(Int64, n_trials/2)], 
			μ_a, μ_ρ; feedback_common = repeat([fc], nb)) for nb in n_blocks] for fc in feedback_commons]
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
	f_seq = Figure(size = (350, 230) .* 72 ./ 25.4)

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
			ylabel = rem(i, plot_ncol) == 1 ? "Fisher information\nper block" : ""
		)

		# Normalize for color
		y_normalized = (med_FI[i] .- minimum(med_FI[i], dims=2)) ./ (maximum(med_FI[i], dims=2) .- minimum(med_FI[i], dims=2))

		# Plot medians
		for j in 1:size(med_FI[i], 2)
			lines!(sum_ax,
				n_blocks,
				med_FI[i][:, j] ./ n_blocks,
				color = y_normalized[:, j],
				colormap = UCL_gradient,
				linewidth = 0.5
			)
		end

	end

	# Compute Spearman's correlations across sections
	cor_avg_rank_med_FI = rotr90(cor(hcat([avg_over(hcat([tiedrank(row) for row in eachrow(matrix)]...), dims = (2,)) for matrix in med_FI]...)))

	ax_corr = Axis(f_seq[row_n, col_n + 1],
		yticks = (1:ns, string.(ns:-1:1)),
		subtitle = "Correlation of avg. ranks\nacross panels",
		subtitlesize = 22)
	hmap = heatmap!(ax_corr, cor_avg_rank_med_FI, colormap = UCL_gradient, 
		colorrange = (-1, 1))

	for i in 1:ns, j in 1:ns
		txtcolor = cor_avg_rank_med_FI[i, j] > 0.5 ? :white : :black
		text!(ax_corr, "$(round(cor_avg_rank_med_FI[i,j], digits = 2))", 
			position = (i, j),
			color = txtcolor, align = (:center, :center), fontsize = 18)
	end

	save("results/sequences.pdf", f_seq, pt_per_unit = 1)

	f_seq	
end

# ╔═╡ Cell order:
# ╠═5d9f27b4-2caf-11ef-3bee-51c614491e13
# ╠═46595874-4cd8-4500-846f-93afbf3d264f
# ╠═9022f0c9-16a4-40eb-81e8-eed9f2c03cc4
# ╠═98b42ee5-f343-4b3f-af52-a2e3fff53409
# ╠═273f93e0-208f-4dd1-ba7d-2d14bc5d2090
# ╠═7554ea02-c54b-4fb1-9d07-79c49c891a80
