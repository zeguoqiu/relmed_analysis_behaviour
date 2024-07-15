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
	# Filter by session
	sess1_data = filter(x -> x.session == "1", PLT_data)

	# Sort
	sort!(sess1_data, [:condition, :prolific_pid, :block, :trial])

	# Make numeric pid variable
	pids = DataFrame(prolific_pid = unique(sess1_data.prolific_pid))
	pids.pp = 1:nrow(pids)

	sess1_data = innerjoin(sess1_data, pids, on = :prolific_pid)

	# Sort
	sort!(sess1_data, [:pp, :block, :trial])

	@assert maximum(sess1_data.block) == 24 "Block numbers are not what you expect"

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

	@assert length(initV(sess1_data)) == nrow(unique(sess1_data[!, [:prolific_pid, :block]])) "initV does not return a vector with length n_total_blocks"

	

end

# ╔═╡ bf5ca997-df1f-4ef3-bf06-1be4e10ff354
to_standata(sess1_data,
			initV;
			model_name = "group_QLrs")

# ╔═╡ f5479de1-4c81-487f-aadb-1e3b07317a02
sess1_data

# ╔═╡ 78549c5d-48b4-4634-b380-b2b8d883d430
begin
	m1_sum, m1_draws, m1_time = load_run_cmdstanr(
		"m1",
		"group_QLrs.stan",
		to_standata(sess1_data,
			initV;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	m1_sum, m1_time
end

# ╔═╡ ccc9fc41-c7af-407a-bfd6-9cc747b2b834


# ╔═╡ 86cbc10b-1fa5-4d23-8154-538106251e28
begin
	t = "odd"
			tdata = filter(x -> t == "odd" ? isodd(x.block) : iseven(x.block), 
			sess1_data)

		tdata.block_new = (tdata.block .+ (t == "odd")) .÷ 2

			to_standata(tdata,
				initV;
				model_name = "group_QLrs",
				block_col = :block_new)
end

# ╔═╡ d3aee72b-8d6b-4a63-a788-2e5f91b1f67e
# Model split half
odd_even_draws = let

	draws = Dict()

	for (i, t) in enumerate(["odd", "even"])
		tdata = filter(x -> t == "odd" ? isodd(x.block) : iseven(x.block), 
			sess1_data)

		tdata.block_new = (tdata.block .+ (t == "odd")) .÷ 2
		
		m1_sum, m1_draws, m1_time = load_run_cmdstanr(
			"m1_$t",
			"group_QLrs.stan",
			to_standata(tdata,
				initV;
				model_name = "group_QLrs",
				block_col = :block_new);
			print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
			threads_per_chain = 3,
			load_model = true
		)
		
		@info m1_sum
		@info "Running time $m1_time minutes."

		draws[t] = m1_draws
	end

	draws
end

# ╔═╡ 76ca2319-9ae5-463e-a53d-47d14373bf87
begin

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
	f_split_half = Figure()

	ax_rho = Axis(f_split_half[1,1],
		xlabel = "Odd blocks reward sensitivity",
		ylabel = "Even blocks reward sensitivity",
		subtitle = "r=$(round(
			cor(rho_split_half.odd, rho_split_half.even), digits= 2))"
	)

	# Regression line
	rho_reg = regression_line_func(rho_split_half, :odd, :even)
	lines!(
		ax_rho,
		range_regression_line(rho_split_half.odd),
		rho_reg.(range_regression_line(rho_split_half.odd)),
		color = :grey,
		linewidth = 4
	)

	# Scatter
	scatter!(
		ax_rho,
		rho_split_half.odd,
		rho_split_half.even,
		markersize = 6
	)

	ax_a = Axis(f_split_half[1,2],
		xlabel = "Odd blocks learning rate",
		ylabel = "Even blocks learning rate",
		subtitle = "r=$(round(
			cor(a_split_half.odd, a_split_half.even), digits= 2))"
	)

	# Regression line
	a_reg = regression_line_func(a_split_half, :odd, :even)
	lines!(
		ax_a,
		range_regression_line(a_split_half.odd) |> a2α,
		a_reg.(range_regression_line(a_split_half.odd)) |> a2α,
		color = :grey,
		linewidth = 4
	)

	scatter!(
		ax_a,
		a2α.(a_split_half.odd),
		a2α.(a_split_half.even),
		markersize = 6
	)

	f_split_half
	
	
end

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
# ╠═bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
# ╠═963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
# ╠═fe070ddf-82cd-4c5f-8bb1-8adab53f654f
# ╠═bf5ca997-df1f-4ef3-bf06-1be4e10ff354
# ╠═f5479de1-4c81-487f-aadb-1e3b07317a02
# ╠═78549c5d-48b4-4634-b380-b2b8d883d430
# ╠═ccc9fc41-c7af-407a-bfd6-9cc747b2b834
# ╠═86cbc10b-1fa5-4d23-8154-538106251e28
# ╠═d3aee72b-8d6b-4a63-a788-2e5f91b1f67e
# ╠═76ca2319-9ae5-463e-a53d-47d14373bf87
