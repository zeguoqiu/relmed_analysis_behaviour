### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ ed4ee070-530e-11ef-143e-09c737b06d9b
begin
	cd("/home/jovyan")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("$(pwd())/relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV, Dates, JSON, RCall, HTTP
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations

	include("$(pwd())/fetch_preprocess_data.jl")
	include("$(pwd())/stan_functions.jl")
	include("$(pwd())/PLT_task_functions.jl")
	include("$(pwd())/fisher_information_functions.jl")
	include("$(pwd())/plotting_functions.jl")

end

# ╔═╡ 0136c52b-019f-490a-87b5-487d2a91414f
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

# ╔═╡ 1fdd69b2-8b02-428a-a7a5-3e7c76656cf5
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ b64cabeb-f82b-4ce7-af56-63c8f02c44de
# Functions needed for data fit
begin
	# Prepare data for q learning model
	function prepare_data_for_fit(data::DataFrame)

		# Sort
		forfit = sort(data, [:condition, :prolific_pid, :block, :trial])
	
		# Make numeric pid variable
		pids = DataFrame(prolific_pid = unique(forfit.prolific_pid))
		pids.pp = 1:nrow(pids)
	
		forfit = innerjoin(forfit, pids, on = :prolific_pid)
	
		# Sort. If more than one session - renumber blocks
		if length(unique(data.session)) > 1
			forfit.cblock = (parse.(Int64, forfit.session) .- 1) .* 
				maximum(forfit.block) .+ forfit.block
			sort!(forfit, [:pp, :cblock, :trial])	
		else
			sort!(forfit, [:pp, :block, :trial])	
		end
		
		
		@assert all(combine(groupby(forfit, [:prolific_pid, :session, :block]),
			:trial => issorted => :trial_sorted
			).trial_sorted) "Trials not sorted"

		@assert all(combine(groupby(forfit, [:prolific_pid, :session]),
			:block => issorted => :block_sorted
		).block_sorted)  "Blocks not sorted"

		return forfit, pids

	end

	aao = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])])
end

# ╔═╡ 91759c41-a311-4861-9868-6665b31b88da
md"""
# Full dataset test-retest
"""

# ╔═╡ ee0d1233-79ee-4c0a-a360-09c9b4b3ed94
function fit_subset(
	data::DataFrame,
	name::String;
	filter_func::Function = x -> true,
	model::String = "group_QLrs02.stan",
	aao::Float64 = aao
)

	# Filter by session
	tdata = filter(filter_func, data)

	# Prepare
	tdata_forfit, tdata_pids = prepare_data_for_fit(tdata)

	m_sum, m_draws, m_trime =  load_run_cmdstanr(
		name,
		"group_QLrs02.stan",
		to_standata(tdata_forfit,
			aao);
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 11,
		parallel_chains = 1,
		iter_sampling = 500
	)

	return m_sum, m_draws, m_trime, tdata_pids

end

# ╔═╡ 521f1a26-1720-42e8-b3be-8e96781eb376
function ppc_suite_fit_models(
	model_name::String,
	data::DataFrame,
	init::Float64
)
	# List of models to fit, and the function creating the releavant data subset
	mods = Dict(
		"full" => x -> true,
		"sess1" => x -> x.session == "1",
		"sess2" => x -> x.session == "2",
		"sess1_odd" => x -> (x.session == "1") & isodd(x.block),
		"sess1_even" => x -> (x.session == "1") & iseven(x.block),
		"sess2_odd" => x -> (x.session == "2") & isodd(x.block),
		"sess2_even" => x -> (x.session == "2") & iseven(x.block)
	)

	# Dict to keep fit results
	fits = Dics()

	for (mod, filter_func) in mods
		fits[mod] = fit_subset(
			data,
			"$(model_name)_$mod",
			filter_func = filter_func,
			model = model,
			aao = init
		)
	end

	return fits
end

# ╔═╡ 63fce3f1-19db-49cf-9cb7-c5695e77f161
function ppc_suite(
	model_name::String,
	data::DataFrame;
	init::Float64 = aao
)

	# Fit models -----------------
	fits = ppc_suite_fit_models(model_name, data, init)

	# PPC by session
	

	# PPC by participants and session

	# Participant parameter spreads

	# Test re-test reliability

	# Split half reliability

	# Reliability by block and trial

	# Compute ICC reliability

	# Return LOO, ICC

end

# ╔═╡ d05df98e-2717-4f24-bfc6-4af22885e276
# Fit both sessions all conditions, aao = aao
begin
	m1s1_sum, m1s1_draws, m1s1_time, m1s1_pids = fit_subset(
		PLT_data,
		"sess1_QL_init_aao",
		filter_func = x -> x.session == "1"
	)

	m1s2_sum, m1s2_draws, m1s2_time, m1s2_pids = fit_subset(
		PLT_data,
		"sess2_QL_init_aao",
		filter_func = x -> x.session == "2"
	)
	
end

# ╔═╡ 225e8909-b044-4ab6-80ec-b34a34b07b2c
# Split-half
begin
	m1s1odd_sum, m1s1odd_draws, m1s1odd_time, m1s1odd_pids = fit_subset(
		PLT_data,
		"sess1_QL_init_aao_odd",
		filter_func = x -> (x.session == "1") & isodd(x.block)
	)

	m1s1even_sum, m1s1even_draws, m1s1even_time, m1s1even_pids = fit_subset(
		PLT_data,
		"sess1_QL_init_aao_even",
		filter_func = x -> (x.session == "1") & iseven(x.block)
	)
	
end

# ╔═╡ 8939f2f2-9293-4780-b4c5-67584f9fe4a8
function reliability_scatter(
	draws1::DataFrame,
	draws2::DataFrame,
	pids1::DataFrame,
	pids2::DataFrame,
	label1::String,
	label2::String
)

	function combine_dfs(
		draws1::DataFrame,
		draws2::DataFrame,
		pids1::DataFrame,
		pids2::DataFrame;
		param::String
	)
			
		p1 = sum_p_params(draws1, param)[!, [:pp, :median]] |>
			x -> rename(x, :median => :m1)
	
		p1 = innerjoin(p1, pids1, on = :pp)
	
		p2 = sum_p_params(draws2, param)[!, [:pp, :median]] |>
			x -> rename(x, :median => :m2)
	
		p2 = innerjoin(p2, pids2, on = :pp)
	
		p1p2 = innerjoin(
			p1[!, Not(:pp)],
			p2[!, Not(:pp)],
			on = :prolific_pid
		)

		return p1p2
	end

	rho1rho2 = combine_dfs(
		draws1,
		draws2,
		pids1,
		pids2;
		param = "rho"
	)

	a1a2 = combine_dfs(
		draws1,
		draws2,
		pids1,
		pids2;
		param = "a"
	)

	# Plot -----------------------------------
	f = Figure(size = (800, 400))

	scatter_regression_line!(
		f[1,1],
		rho1rho2,
		:m1,
		:m2,
		"$label1 reawrd sensitivity",
		"$label2 reawrd sensitivity"
	)

	scatter_regression_line!(
		f[1,2],
		a1a2,
		:m1,
		:m2,
		"$label1 learning rate",
		"$label2 learning rate";
		transform_x = a2α,
		transform_y = a2α
	)


	return f
end

# ╔═╡ c265528e-cdf7-4dc5-ad29-da302ceeeeb6
reliability_scatter(
	m1s1_draws,
	m1s2_draws,
	m1s1_pids,
	m1s2_pids,
	"Session 1",
	"Session 2"
)

# ╔═╡ a12cff45-fe95-48f6-89f8-cf77a8607319
reliability_scatter(
	m1s1odd_draws,
	m1s1even_draws,
	m1s1odd_pids,
	m1s1even_pids,
	"Odd blocks",
	"Even blocks"
)

# ╔═╡ Cell order:
# ╠═ed4ee070-530e-11ef-143e-09c737b06d9b
# ╠═0136c52b-019f-490a-87b5-487d2a91414f
# ╠═1fdd69b2-8b02-428a-a7a5-3e7c76656cf5
# ╠═b64cabeb-f82b-4ce7-af56-63c8f02c44de
# ╟─91759c41-a311-4861-9868-6665b31b88da
# ╠═521f1a26-1720-42e8-b3be-8e96781eb376
# ╠═63fce3f1-19db-49cf-9cb7-c5695e77f161
# ╠═ee0d1233-79ee-4c0a-a360-09c9b4b3ed94
# ╠═d05df98e-2717-4f24-bfc6-4af22885e276
# ╠═c265528e-cdf7-4dc5-ad29-da302ceeeeb6
# ╠═225e8909-b044-4ab6-80ec-b34a34b07b2c
# ╠═a12cff45-fe95-48f6-89f8-cf77a8607319
# ╠═8939f2f2-9293-4780-b4c5-67584f9fe4a8
