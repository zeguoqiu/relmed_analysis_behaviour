### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 74c8335c-4095-11ef-21d3-0715bde378a8
begin
	cd("/home/jovyan")
	import Pkg
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP, JLD2

	include("fetch_preprocess_data.jl")
	include("plotting_functions.jl")
end

# ╔═╡ fb5e4cda-5cdd-492a-8ca2-38fc3fc68ce9
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

# ╔═╡ 1b7c9fc7-af54-4e2f-8303-b64ddd519453
# Load data
begin
	PLT_data = load_PLT_data()
	nothing
end

# ╔═╡ 95b7d6fa-b85f-4095-b53d-afb5fabe9095
md"""
# Data plots from first pilot of PLT
This notebook contains plot of raw data from the fist PLT pilot.

## Overall accuracy curve
"""

# ╔═╡ d97ba043-122c-47b8-ab3e-b3d157f47f42
# Plot overall accuracy
begin

	f_acc = Figure()

	plot_group_accuracy!(f_acc[1,1], PLT_data)

	f_acc

end

# ╔═╡ fadee7a8-cffd-4d8f-b7cc-54ba09dffb50
md"""
## Accuracy curve by session
The substantial difference between sessions is due to a mistake in the experimental design. The order of the feedback sequences was not randomized, resulting in all blocks in early confusing feedback being assigned to the second session.
"""

# ╔═╡ c6ce2aee-24d4-49f8-a57c-2b4e9a3ca022
# Plot by session
begin

	f_sess = Figure()

	plot_group_accuracy!(f_sess[1,1], PLT_data;
		group = :session,
		legend = Dict("1" => "First", "2" => "Second")
	)

	f_sess

end

# ╔═╡ d9e65244-b9ff-48e1-9fa8-e34be86d0e99
md"""
## Accuracy curve by feedback valence
"""

# ╔═╡ 48d11871-5cd3-40f7-adb8-92db011a5d98
# Plot accuracy by valence
begin
	f_acc_valence = Figure()

	plot_group_accuracy!(f_acc_valence[1,1], PLT_data; 
		group = :valence,
		legend = Dict(1 => "Reward", -1 => "Punishment")
	)

	f_acc_valence

end

# ╔═╡ da0702d9-f1ae-4aef-9e8a-feec738b447b
md"""
## Accuracy cruve by early stopping
"""

# ╔═╡ 44985a70-bd56-4e61-a187-a7911c773457
begin
	f_acc_early_stop = Figure()

	plot_group_accuracy!(f_acc_early_stop[1,1], PLT_data; 
		group = :early_stop,
		legend = Dict(false => "never", true => "5 trials"),
		legend_title = "Stop after"
	)

	f_acc_early_stop

end

# ╔═╡ d8581ec0-c195-4040-8cb2-6b493a544cdb
md"""
### Accuracy curve by reward-punishment order
This plot shows session 1 only, since sessions 1 and 2 have big baseline differences.
"""

# ╔═╡ 3c0fa20c-5543-4c92-92d6-3d4495d2cdf5
begin
	f_interleaved_grouped = Figure()

	axs = []
	for (i, (vg, rf)) in enumerate([(false, missing), (true, true), (true, false)])
		ax = plot_group_accuracy!(f_interleaved_grouped[1,i], 
			filter(x -> (x.valence_grouped .== vg) & (ismissing(rf) .|| x.reward_first == rf) & (x.session == "1"), PLT_data); 
			group = :valence,
			title = ["Interleaved", "Reward first", "Punishment first"][i]
		)
		push!(axs, ax)
	end

	linkyaxes!(axs...)

	Legend(f_interleaved_grouped[0,1:3],
		[LineElement(color = Makie.wong_colors()[i]) for i in 1:2],
		["Punishment", "Reward"],
		framevisible = false,
		tellwidth = false,
		orientation = :horizontal,
	)

	f_interleaved_grouped

end

# ╔═╡ 75162f83-a5f5-44f8-a62c-b0d318604074
function plot_split_by_condition(PLT_data)
	f_all_cond = Figure()

	axs = []
	for (i, (es, vg, rf)) in enumerate([(true, false, missing), (true, true, true), (true, true, false), (false, false, missing), (false, true, true), (false, true, false)])
		ax = plot_group_accuracy!(f_all_cond[div(i-1,3) + 1, rem(i-1, 3) + 1], 
			filter(x -> (x.valence_grouped .== vg) & (ismissing(rf) .|| x.reward_first == rf) & (x.early_stop == es), PLT_data); 
			group = :valence,
			title = i <= 3 ? ["Interleaved", "Reward first", "Punishment first"][i] : "",
			ylabel = rem(i, 3) == 1 ? rich(rich("$(es ? "Early stop" : "Never stop")", font = :bold), "\nProp. optimal chice") : "Prop. optimal chice"
		)
		push!(axs, ax)
	end

	linkyaxes!(axs...)

	Legend(f_all_cond[0,1:3],
		[LineElement(color = Makie.wong_colors()[i]) for i in 1:2],
		["Punishment", "Reward"],
		framevisible = false,
		tellwidth = false,
		orientation = :horizontal,
	)

	f_all_cond
end

# ╔═╡ 28cf8cd4-fdf9-49de-bf02-9f40039a28c8
md"""
## Accuracy curve by early stopping and reward-punishment order
First for session 1
"""

# ╔═╡ 61cffe15-3e9e-447b-8fa7-2cde9a83d906
plot_split_by_condition(filter(x -> x.session == "1", PLT_data))

# ╔═╡ 15ec3792-cb56-42e7-a8dd-9fe835862f62
md"""Then for session 2. Note that condition names (Reward first, Punishment first) relate to the order in session 1. The order was reversed in session 2."""

# ╔═╡ ca6b7a59-242e-44b1-9ef5-85759cfd9f93
plot_split_by_condition(filter(x -> x.session == "2", PLT_data))

# ╔═╡ Cell order:
# ╠═74c8335c-4095-11ef-21d3-0715bde378a8
# ╠═fb5e4cda-5cdd-492a-8ca2-38fc3fc68ce9
# ╠═1b7c9fc7-af54-4e2f-8303-b64ddd519453
# ╟─95b7d6fa-b85f-4095-b53d-afb5fabe9095
# ╠═d97ba043-122c-47b8-ab3e-b3d157f47f42
# ╟─fadee7a8-cffd-4d8f-b7cc-54ba09dffb50
# ╠═c6ce2aee-24d4-49f8-a57c-2b4e9a3ca022
# ╟─d9e65244-b9ff-48e1-9fa8-e34be86d0e99
# ╠═48d11871-5cd3-40f7-adb8-92db011a5d98
# ╟─da0702d9-f1ae-4aef-9e8a-feec738b447b
# ╠═44985a70-bd56-4e61-a187-a7911c773457
# ╟─d8581ec0-c195-4040-8cb2-6b493a544cdb
# ╠═3c0fa20c-5543-4c92-92d6-3d4495d2cdf5
# ╠═75162f83-a5f5-44f8-a62c-b0d318604074
# ╠═28cf8cd4-fdf9-49de-bf02-9f40039a28c8
# ╠═61cffe15-3e9e-447b-8fa7-2cde9a83d906
# ╟─15ec3792-cb56-42e7-a8dd-9fe835862f62
# ╠═ca6b7a59-242e-44b1-9ef5-85759cfd9f93
