### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 8c7452ce-49c8-11ef-2441-d5bcc4726e41
begin
	cd("/home/jovyan/")
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO, JuMP, CSV, Dates, JSON, RCall
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations

	include("fetch_preprocess_data.jl")
	include("stan_functions.jl")
	include("PLT_task_functions.jl")
	include("fisher_information_functions.jl")
	include("plotting_functions.jl")

end

# ╔═╡ f3babe5a-a0e3-4b5d-bc5e-630b460dcd06
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

# ╔═╡ 1645ffed-f945-4cb4-9e26-fa7ec40117aa
# Load and clean data
begin
	PLT_data = load_PLT_data()

	PLT_data = exclude_PLT_sessions(PLT_data)

	nothing
end

# ╔═╡ c3a58ad1-3a58-4689-b308-3d17b5325e22
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

# ╔═╡ 821fe42b-c46d-4cc4-89b4-af23637c01e4
# Fit to all data
begin
	# Prepare
	forfit, pids = prepare_data_for_fit(PLT_data)

	# m1_sum, m1_draws, m1_time = load_run_cmdstanr(
	# 	"m1",
	# 	"group_QLrs02.stan",
	# 	to_standata(forfit,
	# 		initV;
	# 		block_col = :cblock,
	# 		model_name = "group_QLrs");
	# 	print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
	# 	threads_per_chain = 3,
	# 	load_model = true
	# )
	# m1_sum, m1_time
end

# ╔═╡ 183e0f9e-2710-4331-a5c0-25f02bbdb33e
begin
	# Filter by session
	sess1_no_early_data = filter(x -> (x.session == "1") & (!x.early_stop), PLT_data)

	# Prepare
	sess1_no_early_forfit, sess1_no_early_pids =
		prepare_data_for_fit(sess1_no_early_data)

	m1s1ne_sum, m1s1ne_draws, m1s1ne_time = load_run_cmdstanr(
		"m1s1ne",
		"group_QLRs02.stan",
		to_standata(sess1_no_early_forfit,
			aao;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3,
		load_model = true
	)
	m1s1ne_sum, m1s1ne_time
end

# ╔═╡ 58c96d00-3247-4254-bf59-ef403708d9c3
to_standata(sess1_no_early_forfit,
			0.;
			model_name = "group_QLrs")

# ╔═╡ 3d055263-edeb-48d4-9ea2-e076326ee207
begin
	m1s1ne0i_sum, m1s1ne0i_draws, m1s1ne0i_time = load_run_cmdstanr(
		"m1s1ne0i",
		"group_QLRs02.stan",
		to_standata(sess1_no_early_forfit,
			0.;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3,
		load_model = true
	)
	m1s1ne_sum, m1s1ne_time
end

# ╔═╡ eeaaea99-673d-4f34-a633-64955caa971e
function extract_participant_params(
	draws::DataFrame;
	params::Vector{String} = ["a", "rho"],
	rescale::Bool = true
) 

	res = Dict()

	for param in params

		# Select columns in draws DataFrame
		tdraws = select(draws, Regex("$(param)\\[\\d+\\]"))
	
		if rescale
			# Add mean and multiply by SD
			tdraws .*= draws[!, Symbol("sigma_$param")]
			tdraws .+= draws[!, Symbol("mu_$param")]	
		end
	
		rename!(s -> replace(s, Regex("$param\\[(\\d+)\\]") => s"\1"),
			tdraws
			)

		res[param] = tdraws
	end

	return res
end

# ╔═╡ cee9b208-46ad-4e31-92c4-d2bfdbad7ad3
function q_learning_posterior_predictive_draw(i;
	data::DataFrame = copy(forfit),
	draw::DataFrame,
	aao::Float64 = mean([mean([0.01, mean([0.5, 1.])]), mean([1., mean([0.5, 0.01])])]) # Average absolute outcome per block
)
		
	# Keep only needed variables
	task = select(
		data,
		[:pp, :session, :block, :trial, :optimalRight, 
			:outcomeRight, :outcomeLeft, :valence, :early_stop]
	)
			
	# Join data with parameters
	task = leftjoin(task, draw, on = :pp)

	# Rearrange data to conform to option A is optimal
	task.feedback_A = ifelse.(
		task.optimalRight .== 1,
		task.outcomeRight,
		task.outcomeLeft
	)

	task.feedback_B = ifelse.(
		task.optimalRight .== 0,
		task.outcomeRight,
		task.outcomeLeft
	)

	# Compute initial values
	task.initV = task.valence .* aao

	# Stop after variable
	task.stop_after = ifelse.(task.early_stop, 5, missing)

	

	# Convenience function for simulation
	function simulate_grouped_block(grouped_df)
		simulate_block(grouped_df,
			2,
			grouped_df.initV[1] .* grouped_df.ρ[1], # Use average reward modulated by rho as initial Q value
			q_learning_w_rho_update,
			[:α, :ρ],
			softmax_choice_direct,
			Vector{Symbol}();
			stop_after = grouped_df.stop_after[1]
		)
	end

	# Simulate data per participants, block
	grouped_task = groupby(task, [:pp, :session, :block])
	sims = transform(grouped_task, simulate_grouped_block)

	sims.draw .= i

	return sims
end


# ╔═╡ 7d836234-8703-48d0-9c66-f39761a57d65
function q_learning_posterior_predictive(
	data::DataFrame,
	draws::DataFrame
)
	participant_params = extract_participant_params(draws)

	Random.seed!(0)

	ppc = []
	for i in sample(1:nrow(participant_params["a"]), 50)

		# Extract draw
		draw = DataFrame(
			α = a2α.(stack(participant_params["a"][1, :])),
			ρ = stack(participant_params["rho"][1, :])
		)
		draw.pp = 1:nrow(draw)

		# Simulate task
		ppd = q_learning_posterior_predictive_draw(i;
			data = data,
			draw = draw
		)

		push!(ppc, ppd)
	end

	return vcat(ppc...)
end

# ╔═╡ 1fc6a457-cd86-4835-a18a-75db1fe4a469
function plot_q_learning_ppc_accuracy(
	data::DataFrame,
	ppc::DataFrame;
	title::String = ""
)
	
	ppc_sum = combine(
		groupby(ppc, 
			[:pp, :draw, :trial]),
		:choice => (x -> mean(x .== 1)) => :isOptimal
	)

	pc_sum = combine(
		groupby(ppc_sum, [:draw, :trial]),
		:isOptimal => mean => :isOptimal
	)

	ppc_sum = combine(
		groupby(ppc_sum, :trial),
		:isOptimal => median => :m,
		:isOptimal => lb => :lb,
		:isOptimal => llb => :llb,
		:isOptimal => ub => :ub,
		:isOptimal => uub => :uub
	)

	f_acc = Figure()

	# Plot data
	ax_acc = plot_group_accuracy!(f_acc[1,1], data;
		error_band = false
	)

	ax_acc.title = title

	# Plot 
	band!(
		ax_acc,
		ppc_sum.trial,
		ppc_sum.llb,
		ppc_sum.uub,
		color = (Makie.wong_colors()[3], 0.1)
	)

	band!(
		ax_acc,
		ppc_sum.trial,
		ppc_sum.lb,
		ppc_sum.ub,
		color = (Makie.wong_colors()[3], 0.3)
	)

	lines!(
		ax_acc,
		ppc_sum.trial,
		ppc_sum.m,
		color = Makie.wong_colors()[3]
	)

	f_acc

end

# ╔═╡ 0bbcf2ce-4297-4543-8659-94ea07b2e0f7
m1s1ne_ppc = q_learning_posterior_predictive(sess1_no_early_forfit, m1s1ne_draws)

# ╔═╡ e9da9e37-dea2-4ea0-8084-d280aa8e3e95
plot_q_learning_ppc_accuracy(sess1_no_early_forfit, m1s1ne_ppc;
	title = "Initial value = $(round(aao, digits = 3))"
)

# ╔═╡ 7f1f513a-b960-4dd5-b1c5-9f87e4af5ae9
m1s1ne0i_ppc = q_learning_posterior_predictive(sess1_no_early_forfit, m1s1ne0i_draws)

# ╔═╡ 3def17fd-3ac7-416d-bcb0-dba48ec1918b
plot_q_learning_ppc_accuracy(sess1_no_early_forfit, m1s1ne0i_ppc;
	title = "Initial value = 0")

# ╔═╡ dce1e578-055a-45fb-97d1-99842e222068
let
	f_bivar_post = Figure()

	function plot_bivariate_posterior!(f::GridPosition,
		draws::DataFrame;
		x::Symbol,
		y::Symbol,
		xlabel::String,
		ylabel::String,
		transform_x::Function = x -> x,
		transform_y::Function = y -> y
	)
		ax = Axis(
			f,
			xlabel = xlabel,
			ylabel = ylabel
		)
	
		scatter!(
			ax,
			transform_x.(draws[!, x]),
			transform_y.(draws[!, y]),
			alpha = .1,
			markersize = 4
		)
	end

	xs = [:mu_rho, :mu_rho, :mu_a, :sigma_rho, :sigma_rho, :sigma_a]
	ys = [:mu_a, :sigma_a, :sigma_a, :sigma_a, Symbol("rho[1]"), Symbol("a[2]")]
	labs = Dict(
		:mu_rho => "Mean reward sensitivity",
		:mu_a => "Mean learning rate",
		:sigma_rho => "SD reward sensitivity",
		:sigma_a => "SD learningrate",
		Symbol("rho[1]") => "Participant 1\nreward sensitivity",
		Symbol("a[2]") => "Participant 1\nlearning rate"
	)
	xlabs = [labs[x] for x in xs]
	ylabs = [labs[y] for y in ys]
	fxs = [occursin("rho", string(x)) ? (z -> z) : a2α for x in xs]
	fys = [occursin("rho", string(y)) ? (z -> z) : a2α for y in ys]
	
	for (r, c, x, y, xlab, ylab, fx, fy) in zip(
		[1,1,1,2,2,2],
		[1,2,3,1,2,3],
		xs,
		ys,
		xlabs,
		ylabs,
		fxs,
		fys
	)
		plot_bivariate_posterior!(
			f_bivar_post[r, c],
			m1s1ne_draws,
			x = x,
			y = y,
			xlabel = xlab,
			ylabel = ylab,
			transform_x = fx,
			transform_y = fy
		)
	end

	f_bivar_post

end

# ╔═╡ Cell order:
# ╠═8c7452ce-49c8-11ef-2441-d5bcc4726e41
# ╠═f3babe5a-a0e3-4b5d-bc5e-630b460dcd06
# ╠═1645ffed-f945-4cb4-9e26-fa7ec40117aa
# ╠═c3a58ad1-3a58-4689-b308-3d17b5325e22
# ╠═821fe42b-c46d-4cc4-89b4-af23637c01e4
# ╠═183e0f9e-2710-4331-a5c0-25f02bbdb33e
# ╠═58c96d00-3247-4254-bf59-ef403708d9c3
# ╠═3d055263-edeb-48d4-9ea2-e076326ee207
# ╠═eeaaea99-673d-4f34-a633-64955caa971e
# ╠═cee9b208-46ad-4e31-92c4-d2bfdbad7ad3
# ╠═7d836234-8703-48d0-9c66-f39761a57d65
# ╠═1fc6a457-cd86-4835-a18a-75db1fe4a469
# ╠═0bbcf2ce-4297-4543-8659-94ea07b2e0f7
# ╠═e9da9e37-dea2-4ea0-8084-d280aa8e3e95
# ╠═7f1f513a-b960-4dd5-b1c5-9f87e4af5ae9
# ╠═3def17fd-3ac7-416d-bcb0-dba48ec1918b
# ╠═dce1e578-055a-45fb-97d1-99842e222068
