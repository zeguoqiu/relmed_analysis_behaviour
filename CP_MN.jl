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

# ╔═╡ bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
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
	
		@assert maximum(forfit.block) == 24 "Block numbers are not what you expect"
	
		
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
		threads_per_chain = 3,
		load_model = true
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
	write_cor::Bool = true
)

	x = df[!, x_col]
	y = df[!, y_col]
	
	ax = Axis(f,
		xlabel = xlabel,
		ylabel = ylabel,
		subtitle = write_cor ? "r=$(round(
			cor(x, y), digits= 2))" : ""
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

	scatter_regression_line!(
		f_split_half[1,1],
		rho_split_half,
		:odd,
		:even,
		"Odd blocks reward sensitivity",
		"Even blocks reward sensitivity"
	)


	scatter_regression_line!(
		f_split_half[1,2],
		a_split_half,
		:odd,
		:even,
		"Odd blocks learning rate",
		"Even blocks learning rate";
		transform_x = a2α,
		transform_y = a2α
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
		threads_per_chain = 3,
		load_model = true
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


# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
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
