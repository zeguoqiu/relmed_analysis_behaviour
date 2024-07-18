### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ e01188c3-ca30-4a7c-9101-987752139a71
begin
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP, RCall, JLD2, Dates

	include("fetch_preprocess_data.jl")
	include("stan_functions.jl")
	include("plotting_functions.jl")

end

# ╔═╡ b287a29c-17cf-482e-8358-07ae6600c0e6
conference = "BAP"

# ╔═╡ bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
begin
	# Set theme
	inter_bold = assetpath(pwd() * "/fonts/Inter/Inter-Bold.ttf")
	
	cp_th = Theme(
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

	bap_th = Theme(
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

	if conference == "BAP"
		set_theme!(bap_th)
	else
		set_theme!(cp_th)
	end
end

# ╔═╡ 963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

end

# ╔═╡ fe070ddf-82cd-4c5f-8bb1-8adab53f654f
# Session 1 model
begin

	# Function to get initial values
	function initV(data::DataFrame)

		# List of blocks
		blocks = unique(data[!, [:pp, :block, :valence, :valence_grouped]])

		# Absolute mean reward for grouped
		amrg = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
		
		# Valence times amrg
		initVs = blocks.valence .* amrg

		return [fill(i, 2) for i in initVs]
	end

	# Prepare data for q learning model
	function prepare_data_for_fit(data::DataFrame)

		# Sort
		forfit = sort(data, [:condition, :prolific_pid, :block, :trial])
	
		# Make numeric pid variable
		pids = DataFrame(prolific_pid = unique(forfit.prolific_pid))
		pids.pp = 1:nrow(pids)
	
		forfit = innerjoin(forfit, pids, on = :prolific_pid)
	
		# Sort
		sort!(forfit, [:pp, :block, :trial])	
		
		@assert length(initV(forfit)) == 
			nrow(unique(forfit[!, [:prolific_pid, :block]])) "initV does not return a vector with length n_total_blocks"

		return forfit, pids

	end

end

# ╔═╡ 78549c5d-48b4-4634-b380-b2b8d883d430
begin
	# Filter by session
	sess1_data = filter(x -> x.session == "1", PLT_data)

	# Prepare
	sess1_forfit, sess1_pids = prepare_data_for_fit(sess1_data)

	m1s1_sum, m1s1_draws, m1s1_time = load_run_cmdstanr(
		"m1s1",
		"group_QLrs.stan",
		to_standata(sess1_forfit,
			initV;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	m1s1_sum, m1s1_time
end

# ╔═╡ d3aee72b-8d6b-4a63-a788-2e5f91b1f67e
# Model split half
odd_even_draws = let

	draws = Dict()

	for (i, t) in enumerate(["odd", "even"])
		tdata = filter(x -> t == "odd" ? isodd(x.block) : iseven(x.block), 
			sess1_forfit)

		tdata.block_new = (tdata.block .+ (t == "odd")) .÷ 2
		
		m1s1_sum, m1s1_draws, m1s1_time = load_run_cmdstanr(
			"m1s1_$t",
			"group_QLrs.stan",
			to_standata(tdata,
				initV;
				model_name = "group_QLrs",
				block_col = :block_new);
			print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
			threads_per_chain = 3
		)
		
		@info m1s1_sum
		@info "Running time $m1s1_time minutes."

		draws[t] = m1s1_draws
	end

	draws
end

# ╔═╡ 30f821eb-6c91-40cf-b218-44025b1e8904
function scatter_regression_line!(
	f::GridPosition,
	df::DataFrame,
	x_col::Symbol,
	y_col::Symbol,
	xlabel::String,
	ylabel::String;
	transform_x::Function = x -> x,
	transform_y::Function = x -> x,
	color = Makie.wong_colors()[1],
	legend::Union{Dict, Missing} = missing,
	legend_title::String = "",
	write_cor::Bool = true,
	cor_correction::Function = x -> x, # Correction to apply for correlation, e.g. Spearman Brown
	cor_label::String = "r"
)

	x = df[!, x_col]
	y = df[!, y_col]
	
	ax = Axis(f,
		xlabel = xlabel,
		ylabel = ylabel,
		subtitle = write_cor ? "$cor_label=$(round(
			cor_correction(cor(x, y)), digits= 2))" : ""
	)

	# Regression line
	treg = regression_line_func(df, x_col, y_col)
	lines!(
		ax,
		range_regression_line(x) |> transform_x,
		treg.(range_regression_line(x)) |> transform_y,
		color = :grey,
		linewidth = 4
	)

	sc = scatter!(
		ax,
		transform_x.(x),
		transform_y.(y),
		markersize = 6,
		color = color
	)

	if !ismissing(legend)
		Legend(
			f,
			[MarkerElement(color = k, marker = :circle) for k in keys(legend)],
			[legend[k] for k in keys(legend)],
			legend_title,
			halign = :right,
			valign = :top,
			framevisible = false,
			tellwidth = false,
			tellheight = false
		)

	end
end

# ╔═╡ 76ca2319-9ae5-463e-a53d-47d14373bf87
let

	# Split half rho
	rho_odd = sum_p_params(odd_even_draws["odd"], "rho")[!, [:pp, :median]] |>
		x -> rename(x, :median => :odd)

	rho_even = sum_p_params(odd_even_draws["even"], "rho")[!, [:pp, :median]] |>
		x -> rename(x, :median => :even)

	rho_split_half = innerjoin(rho_odd, rho_even, on = :pp)

	# Split half alpha
	a_odd = sum_p_params(odd_even_draws["odd"], "a")[!, [:pp, :median]] |>
		x -> rename(x, :median => :odd)

	a_even = sum_p_params(odd_even_draws["even"], "a")[!, [:pp, :median]] |>
		x -> rename(x, :median => :even)

	a_split_half = innerjoin(a_even, a_odd, on = :pp)

	# Plot -------------------------------------
	f_split_half = Figure(size = (800, 400))

	spearman_brown(r) = (2 * r) / (1 + r)

	scatter_regression_line!(
		f_split_half[1,1],
		rho_split_half,
		:odd,
		:even,
		"Odd blocks reward sensitivity",
		"Even blocks reward sensitivity";
		cor_correction = spearman_brown,
		cor_label = "Spearman-Brown r*"
	)


	scatter_regression_line!(
		f_split_half[1,2],
		a_split_half,
		:odd,
		:even,
		"Odd blocks learning rate",
		"Even blocks learning rate";
		transform_x = a2α,
		transform_y = a2α,
		cor_correction = spearman_brown,
		cor_label = "Spearman-Brown r*"
	)

	save("results/split_half_scatters.png", f_split_half, pt_per_unit = 1)
	save("results/split_half_scatters.pdf", f_split_half, pt_per_unit = 1)
	
	f_split_half
end

# ╔═╡ 844bcefd-81a4-489f-9a44-534253553bf2
begin
	# Filter by session
	sess2_data = filter(x -> x.session == "2", PLT_data)

	# Prepare
	sess2_forfit, sess2_pids = prepare_data_for_fit(sess2_data)

	m1s2_sum, m1s2_draws, m1s2_time = load_run_cmdstanr(
		"m1s2",
		"group_QLrs.stan",
		to_standata(sess2_forfit,
			initV;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	m1s2_sum, m1s2_time
end

# ╔═╡ d3bc8bba-e2b0-4399-9eb7-bfc10b8f65ae
let
	# Test-retest rho
	rho_sess1 = sum_p_params(m1s1_draws, "rho")[!, [:pp, :median]] |>
		x -> rename(x, :median => :sess1)

	rho_sess1 = innerjoin(rho_sess1, sess1_pids, on = :pp)

	rho_sess2 = sum_p_params(m1s2_draws, "rho")[!, [:pp, :median]] |>
		x -> rename(x, :median => :sess2)

	rho_sess2 = innerjoin(rho_sess2, sess2_pids, on = :pp)

	rho_retest = innerjoin(
		rho_sess1[!, Not(:pp)],
		rho_sess2[!, Not(:pp)],
		on = :prolific_pid
	)

	# Test-retest a
	a_sess1 = sum_p_params(m1s1_draws, "a")[!, [:pp, :median]] |>
		x -> rename(x, :median => :sess1)

	a_sess1 = innerjoin(a_sess1, sess1_pids, on = :pp)

	a_sess2 = sum_p_params(m1s2_draws, "a")[!, [:pp, :median]] |>
		x -> rename(x, :median => :sess2)

	a_sess2 = innerjoin(a_sess2, sess2_pids, on = :pp)

	a_retest = innerjoin(
		a_sess1[!, Not(:pp)],
		a_sess2[!, Not(:pp)],
		on = :prolific_pid
	)

	# Plot -----------------------------------
	f_retest = Figure(size = (800, 400))

	scatter_regression_line!(
		f_retest[1,1],
		rho_retest,
		:sess1,
		:sess2,
		"Session 1 reward sensitivity",
		"Session 2 reward sensitivity"
	)

	scatter_regression_line!(
		f_retest[1,2],
		a_retest,
		:sess1,
		:sess2,
		"Session 1 learning rate",
		"Session 2 learning rate";
		transform_x = a2α,
		transform_y = a2α
	)

	save("results/test_retest_scatters.pdf", f_retest, pt_per_unit = 1)
	save("results/test_retest_scatters.png", f_retest, pt_per_unit = 1)


	f_retest

end

# ╔═╡ a7d7e648-6cb0-4e2c-a4f9-f951a61e3f20
# Correalation between parameters
let
	rho = sum_p_params(m1s1_draws, "rho")[!, [:pp, :median]] |>
		x -> rename(x, :median => :rho)

	a = sum_p_params(m1s1_draws, "a")[!, [:pp, :median]] |>
		x -> rename(x, :median => :a)

	bivariate_post = innerjoin(rho, a, on = :pp)

	bivariate_post = innerjoin(bivariate_post, sess1_pids, on = :pp)

	bivariate_post = innerjoin(bivariate_post, 
		unique(sess1_forfit[!, 
			[:prolific_pid, :early_stop, :reward_first, :valence_grouped]]),
		on = :prolific_pid)

	# Plot ----------------
	f_bivar = Figure(size = (800, 800))

	scatter_regression_line!(
		f_bivar[1,1],
		bivariate_post,
		:rho,
		:a,
		"Reward sensitivity",
		"Learning rate",
		transform_y = a2α
	)

	scatter_regression_line!(
		f_bivar[1,2],
		bivariate_post,
		:rho,
		:a,
		"Reward sensitivity",
		"Learning rate",
		transform_y = a2α,
		color = Makie.wong_colors()[bivariate_post.early_stop .+ 1],
		legend = Dict(Makie.wong_colors()[1] => "Full block", Makie.wong_colors()[2] => "Early stop"),
		write_cor = false
	)

	scatter_regression_line!(
		f_bivar[2,1],
		bivariate_post,
		:rho,
		:a,
		"Reward sensitivity",
		"Learning rate",
		transform_y = a2α,
		color = Makie.wong_colors()[bivariate_post.valence_grouped .+ 1],
		legend = Dict(Makie.wong_colors()[1] => "Interleaved", Makie.wong_colors()[2] => "Grouped"),
		legend_title = "Valence",
		write_cor = false
	)

	scatter_regression_line!(
		f_bivar[2,2],
		bivariate_post,
		:rho,
		:a,
		"Reward sensitivity",
		"Learning rate",
		transform_y = a2α,
		color = Makie.wong_colors()[bivariate_post.valence_grouped .+ 1 + bivariate_post.reward_first],
		legend = Dict(
			Makie.wong_colors()[1] => "Interleaved", 
			Makie.wong_colors()[2] => "Punishment first",
			Makie.wong_colors()[3] => "Reward first"),
		legend_title = "Valence",
		write_cor = false
	)

	f_bivar
	
end

# ╔═╡ ce562d2b-0894-4a29-9bf9-3f45342bd057
incremental_draws_sess1, incremental_draws_sess2 = let

	sess1_draws = []
	sess2_draws = []

	for s in 1:2
		for i in 1:24
			# Filter data
			tdata = filter(x -> (x.session == "$s") &
				(x.block <= i), PLT_data)
		
			# Prepare
			tforfit, tpids = prepare_data_for_fit(tdata)
		
			m1_sum, m1_draws, m1_time = load_run_cmdstanr(
				i < 24 ? "m1s$(s)b$i" : "m1s$(s)",
				"group_QLrs.stan",
				to_standata(tforfit,
					initV;
					model_name = "group_QLrs");
				print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
				threads_per_chain = 3
			)
	
			push!(s == 1 ? sess1_draws : sess2_draws, m1_draws)

			@info m1_sum
			@info "Running time $m1_time minutes"
	
		end
	end

	sess1_draws, sess2_draws

end

# ╔═╡ 5b9ca47b-e3c3-4eba-a80b-9cde46201104
# ╠═╡ skip_as_script = true
#=╠═╡
let

	f_n_blocks = Figure(size = (800, 400))

	function covergence_plot(f, incremental_draws, parameter, y_label)

		sum_a_blocks = []
		for (i, d) in enumerate(incremental_draws)
			ts = sum_p_params(d, parameter; transform = false)[!, [:pp, :median]]
	
			ts.n_blocks .= i
	
			push!(sum_a_blocks, ts)
		end
	
		sum_a_blocks = vcat(sum_a_blocks...)
				
		sum_a_blocks = unstack(sum_a_blocks, :pp, :n_blocks, :median)
	
		sum_a_blocks = Matrix(sum_a_blocks[:, Not(:pp)])
	
		sum_a_blocks .-= sum_a_blocks[:, end]
	
	
		ax = Axis(
			f,
			xlabel = "# of blocks",
			ylabel = y_label,
			xautolimitmargin = (0., 0.05f0),
			xticks = round.(range(1, size(sum_a_blocks, 2), 4)[2:4])
		)
	
		for i in 1:size(sum_a_blocks, 1)
			lines!(ax, 
				1:size(sum_a_blocks,2), 
				sum_a_blocks[i, :],
				color = Makie.wong_colors()[1],
				alpha = 0.4
			)
		end
	end

	covergence_plot(f_n_blocks[1,1], incremental_draws_sess1, "rho", "Deviation from final\nreward sensitivity estimate")

	covergence_plot(f_n_blocks[1,2], incremental_draws_sess1, "a", "Deviation from final\nlearning rate estimate")

	save("results/convergence.pdf", f_n_blocks, pt_per_unit = 1)
	save("results/convergence.png", f_n_blocks, pt_per_unit = 1)

	f_n_blocks


end
  ╠═╡ =#

# ╔═╡ 054d61e9-3c78-454d-9f65-e2cb22460395
incremental_trial_draws_sess1, incremental_trial_draws_sess2 = let

	sess1_draws = []
	sess2_draws = []

	for s in 1:2
		for i in 1:13
			# Filter data
			tdata = filter(x -> (x.session == "$s") &
				(x.trial <= i), PLT_data)
		
			# Prepare
			tforfit, tpids = prepare_data_for_fit(tdata)
		
			m1_sum, m1_draws, m1_time = load_run_cmdstanr(
				i < 13 ? "m1s$(s)t$i" : "m1s$(s)",
				"group_QLrs.stan",
				to_standata(tforfit,
					initV;
					model_name = "group_QLrs");
				print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
				threads_per_chain = 3
			)
	
			push!(s == 1 ? sess1_draws : sess2_draws, m1_draws)
	
		end
	end

	sess1_draws, sess2_draws

end

# ╔═╡ 9316eb8f-ff77-4f6f-b5df-5ad2b6d03959
# ╠═╡ skip_as_script = true
#=╠═╡
begin
	# Compute average time for instructions and average time per block in minutes
	instruction_time = mean(filter(
		x -> (x.block == 1) & (x.trial == 1), 
		PLT_data).time_elapsed) / 1000 / 60

	block_time = mean(
		combine(
			groupby(
				PLT_data, [:prolific_pid, :session]
			),
		:time_elapsed => (x -> maximum(x) - minimum(x)) => :duration
	).duration) / 1000 / 24 / 60

	function reliability_plot(ax, 
		incremental_draws_sess1, 
		incremental_draws_sess2, 
		parameter;
		color
	)

		cors = zeros(Float64, 1:length(incremental_draws_sess1))
		for (i, (d1, d2)) in enumerate(zip(incremental_draws_sess1, 
			incremental_draws_sess2))
			
			ts1 = sum_p_params(d1, parameter)[!, [:pp, :median]] |>
				x -> rename(x, :median => :sess1)
		
			ts1 = innerjoin(ts1, sess1_pids, on = :pp)
		
			ts2 = sum_p_params(d2, parameter)[!, [:pp, :median]] |>
				x -> rename(x, :median => :sess2)
		
			ts2 = innerjoin(ts2, sess2_pids, on = :pp)
		
			trt = innerjoin(
				ts1[!, Not(:pp)],
				ts2[!, Not(:pp)],
				on = :prolific_pid
			)
	
			cors[i] = cor(trt.sess1, trt.sess2)
		end	
	
		lines!(ax, 
			1:length(cors), 
			cors,
			linewidth = 3,
			color = color
		)		
	end

	f_reliability_time = Figure(size = (190, 219) .* 72 ./ 25.4)

	# Plot per block
	ax_reliabitiliy_block = Axis(
		f_reliability_time[1,1],
		xlabel = "# of block (task duration)",
		ylabel = "Test-retest reliability",
		xtickformat = values -> 
			["$(round(Int64, value))\n($(round(instruction_time + block_time * value, digits = 2))')" for value in values],
		xautolimitmargin = (0., 0.05f0),
		xticks = 5:5:20
	)

	reliability_plot(
		ax_reliabitiliy_block, 
		incremental_draws_sess1, 
		incremental_draws_sess2, 
		"a",
		color = "#34C6C6"
	)

	reliability_plot(
		ax_reliabitiliy_block, 
		incremental_draws_sess1, 
		incremental_draws_sess2, 
		"rho",
		color = "#FFCA36"
	)

	# Plot per trial
	ax_reliabitiliy_trial = Axis(
		f_reliability_time[2,1],
		xlabel = "# of trials (task duration)",
		ylabel = "Test-retest reliability",
		xtickformat = values -> 
			["$(round(Int64, value))\n($(round(instruction_time + block_time / 13 * 24 * value, digits = 2))')" for value in values],
		xautolimitmargin = (0., 0.05f0)
	)

	reliability_plot(
		ax_reliabitiliy_trial, 
		incremental_trial_draws_sess1, 
		incremental_trial_draws_sess2, 
		"a",
		color = "#34C6C6"
	)

	reliability_plot(
		ax_reliabitiliy_trial, 
		incremental_trial_draws_sess1, 
		incremental_trial_draws_sess2, 
		"rho",
		color = "#FFCA36"
	)

	rowgap!(f_reliability_time.layout, 30)

	save("results/time_reliability.pdf", f_reliability_time, pt_per_unit = 1)
	save("results/time_reliability.png", f_reliability_time, pt_per_unit = 1)

	f_reliability_time
	
end
  ╠═╡ =#

# ╔═╡ 5a152bb7-df44-46c3-ae6f-9fa14922c82c
# Accuracy plot
begin
	f_acc_valence = Figure(size=(190, 142) .* 72 ./ 25.4)


	plot_group_accuracy!(f_acc_valence[1,1], filter(x -> x.condition == "00", sess1_data); 
		group = :valence,
		legend = Dict(1 => "Reward", -1 => "Punishment"),
		levels = [1, -1],
		colors = ["#52C152", "#34C6C6"]
	)

	save("results/accuracy_by_valence.pdf", f_acc_valence, pt_per_unit = 1)
	f_acc_valence

end

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
# ╠═b287a29c-17cf-482e-8358-07ae6600c0e6
# ╠═bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
# ╠═963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
# ╠═fe070ddf-82cd-4c5f-8bb1-8adab53f654f
# ╠═78549c5d-48b4-4634-b380-b2b8d883d430
# ╠═d3aee72b-8d6b-4a63-a788-2e5f91b1f67e
# ╠═30f821eb-6c91-40cf-b218-44025b1e8904
# ╠═76ca2319-9ae5-463e-a53d-47d14373bf87
# ╠═844bcefd-81a4-489f-9a44-534253553bf2
# ╠═d3bc8bba-e2b0-4399-9eb7-bfc10b8f65ae
# ╠═a7d7e648-6cb0-4e2c-a4f9-f951a61e3f20
# ╠═ce562d2b-0894-4a29-9bf9-3f45342bd057
# ╠═5b9ca47b-e3c3-4eba-a80b-9cde46201104
# ╠═054d61e9-3c78-454d-9f65-e2cb22460395
# ╠═9316eb8f-ff77-4f6f-b5df-5ad2b6d03959
# ╠═5a152bb7-df44-46c3-ae6f-9fa14922c82c
