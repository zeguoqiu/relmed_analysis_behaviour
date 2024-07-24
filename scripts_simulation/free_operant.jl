### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 16cc5376-33b1-11ef-05af-69d6f8d63512
begin
	cd("/home/jovyan")
    import Pkg
    # activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase,
		ForwardDiff, LinearAlgebra, Memoization, LRUCache, GLM, JLD2, FileIO
	using IterTools: product
	using LogExpFunctions: logsumexp
	using Combinatorics: combinations
end

# ╔═╡ 1f0aca6b-bfee-4f4d-86ff-4d318037f20f
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


# ╔═╡ 6b17bbab-f812-45de-91e6-1823a1b158c2
function plot_free_operant_formula(
	model::Function;
	intercept::Vector{Float64},
	slope::Vector{Float64},
	res::Int64 = 200,
	Δ::Vector{Int64} = [res-40, res],
	slope_label::Union{String, Makie.RichText} = "β",
	intercept_label::Union{String, Makie.RichText} = "α",
	title = ""
)

	# Set up figure
	f = Figure()
	ax = Axis(
		f[1,1],
		xlabel = "R̄ - average reward per action",
		ylabel = "λ - expected action per τ",
		title = title
	)

	# Plot lines
	x = range(0.01, 10., 200)
	for (a, b) in zip(intercept, slope)
		y = model.(a, b, x)
		
		lines!(ax,
			x,
			y
		)

		bracket!(
			ax,
			x[Δ[1]],
			y[Δ[1]],
			x[Δ[2]],
			y[Δ[2]];
			text = rich(slope_label, "=$b, ", intercept_label, "=$a"),
			width = 0,
			linewidth = 0,
		)
	end

	return f
end
	

# ╔═╡ 7cd0f7b6-c3e0-4af8-bb99-6da71ea4789d
let
	niv_model(α, C_v, R) = sqrt((R + α) / C_v)

	plot_free_operant_formula(
		niv_model,
		intercept = [0., 0., 5.],
		slope = [0.5, .7, 0.5],
		slope_label = rich("C", subscript("v")),
		title = L"\lambda = \sqrt{\frac{\alpha + \bar{R}}{C_v}}"
	)

end

# ╔═╡ 0a66c6cd-b199-40d2-bc32-582f73e339ed
let
	logliner_model(α, β, R) = exp(α + β * R)

	plot_free_operant_formula(
		logliner_model,
		intercept = [0.5, 0.5, 0.6],
		slope = [0.5, 0.6, 0.5],
		# Δ=[40, 60],
		title = L"log(\lambda) = \alpha + \beta \bar{R}"
	)

end

# ╔═╡ bc16cda4-e392-4217-a8d6-d26e34bf5f4b
let
	logsqrt_model(α, β, R) = exp(α + β * sqrt(R))

	plot_free_operant_formula(
		logsqrt_model,
		intercept = [0.5, 0.5, 1.],
		slope = [0.5, 0.6, 0.5],
		title = L"log(\lambda) = \alpha + \beta \sqrt{\bar{R}}"
	)

end

# ╔═╡ ecb833e2-cad6-44b3-a570-905e9e36aaa1
function niv_agent(
	α::Float64, # Intercept - response rate for no reward
	C_v::Float64, # Vigour cost
	R_bar::Float64; # Average reward per action
	τ::Float64 = 1.0 # Time period - multiplies all
)

	return rand(Poisson(τ * sqrt((R_bar + α ) / C_v)))

end

# ╔═╡ a4ed62da-e666-4879-822f-8605a587abfa
replicate_elements(v, k) = vcat([fill(v[i], div(k, length(v)) + (i <= mod(k, length(v)))) for i in 1:length(v)]...)

# ╔═╡ e9878079-ebe3-4b78-ab6d-6cb92b1f6431
function free_operant_task_structure(
	n_blocks::Int64,
	n_trials::Int64, # Per block
	exchange_rates::Vector{Float64}, # Exchange rates - this should be of length n_block, and will multiply rewards
	rewards::Vector{Vector{Float64}} # Rewards, to be multiplied by exchange rate. Should be of length n_block. If each element is of leanght n_trials - use as is. Otherwise, sample randomly.
	)

	@assert length(exchange_rates) == n_blocks

	@assert length(rewards) == n_blocks

	if !all(length.(rewards) .== n_trials)
		scaled_rewards = vcat([shuffle(replicate_elements(shuffle(rewards[b]), 
			n_trials)) * exchange_rates[b] for b in 1:n_blocks]...) 
	else
		scaled_rewards = vcat([r * 
			exchange_rates[b] for (b,r) in enumrate(rewards)]...) 
	end

	block_n = vcat([fill(b, n_trials) for b in 1:n_blocks]...)

	trial_n = repeat(1:n_trials, n_blocks)

	return (block = block_n, trial = trial_n, reward = scaled_rewards)
end

# ╔═╡ 493918f8-3d36-421a-8b6d-c34255d076e0
function add_params(n_blocks, n_trials, exchange_rates, rewards, α, C_v)
	task = free_operant_task_structure(
		n_blocks, 
		n_trials, 
		exchange_rates, 
		rewards
	)

	t

end

# ╔═╡ 62c3de2c-7284-44e0-a3c8-b664124dc7c5
let task = free_operant_task_structure(4, 12, repeat([4., 0.5], 2), fill(collect(range(0.01, 2.5, 6)), 4))
	
end

# ╔═╡ Cell order:
# ╠═16cc5376-33b1-11ef-05af-69d6f8d63512
# ╠═1f0aca6b-bfee-4f4d-86ff-4d318037f20f
# ╠═7cd0f7b6-c3e0-4af8-bb99-6da71ea4789d
# ╠═0a66c6cd-b199-40d2-bc32-582f73e339ed
# ╠═bc16cda4-e392-4217-a8d6-d26e34bf5f4b
# ╠═6b17bbab-f812-45de-91e6-1823a1b158c2
# ╠═ecb833e2-cad6-44b3-a570-905e9e36aaa1
# ╠═a4ed62da-e666-4879-822f-8605a587abfa
# ╠═e9878079-ebe3-4b78-ab6d-6cb92b1f6431
# ╠═493918f8-3d36-421a-8b6d-c34255d076e0
# ╠═62c3de2c-7284-44e0-a3c8-b664124dc7c5
