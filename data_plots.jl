### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 74c8335c-4095-11ef-21d3-0715bde378a8
begin
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP, JLD2

	include("fetch_preprocess_data.jl")
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
PLT_data = load_PLT_data()

# ╔═╡ 1af9e9df-0d5c-4f7b-a14f-9fc69a5e4d9e
function plot_group_accuracy!(
	f::GridPosition,
	data::DataFrame;
	group::Union{Symbol, Missing} = missing,
	colors = Makie.wong_colors(),
	title::String = "",
	legend::Union{Dict, Missing} = missing,
	legend_title::String = "",
	backgroundcolor = :white,
	ylabel::Union{String, Makie.RichText}="Prop. optimal chioce"
	)

	# Default group value
	tdata = copy(data)
	if ismissing(group)
		tdata.group .= 1
		group = :group
	else
		tdata.group = tdata[!, group]
	end


	# Summarize into proportion of participants choosing optimal
	sum_data = combine(
		groupby(tdata, [:prolific_pid, :group, :trial]),
		:isOptimal => mean => :acc
	)

	sum_data = combine(
		groupby(sum_data, [:group, :trial]),
		:acc => mean => :acc,
		:acc => sem => :acc_sem
	)

	# Set up axis
	ax = Axis(f[1,1],
		xlabel = "Trial #",
		ylabel = ylabel,
		xautolimitmargin = (0., 0.),
		xticks = range(1, round(Int64, maximum(sum_data.trial)), 4),
		backgroundcolor = backgroundcolor,
		title = title
	)

	group_levels = unique(sum_data.group)
	for (i,g) in enumerate(group_levels)
		gdat = filter(:group => (x -> x==g), sum_data)

		# Plot line
		band!(ax,
			gdat.trial,
			gdat.acc - gdat.acc_sem,
			gdat.acc + gdat.acc_sem,
			color = (colors[i], 0.3)
		)
		
		lines!(ax, 
			gdat.trial, 
			gdat.acc, 
			color = colors[i],
			linewidth = 3)
	end

	if !ismissing(legend)
		elements = [LineElement(color = colors[i]) for i in 1:length(group_levels)]
		labels = [legend[g] for g in group_levels]
		
		Legend(f[0,1],
			elements,
			labels,
			legend_title,
			framevisible = false,
			tellwidth = false,
			orientation = :horizontal,
			titleposition = :left
		)
		# rowsize!(f.layout, 0, Relative(0.1))
	end
		

	return ax

end

# ╔═╡ d97ba043-122c-47b8-ab3e-b3d157f47f42
# Plot overall accuracy
begin

	f_acc = Figure()

	plot_group_accuracy!(f_acc[1,1], PLT_data)

	f_acc

end

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

# ╔═╡ bfd2018a-b45c-4eba-b898-39b2a5f10134
unique(PLT_data.valence_grouped)

# ╔═╡ 3c0fa20c-5543-4c92-92d6-3d4495d2cdf5
begin
	f_interleaved_grouped = Figure()

	axs = []
	for (i, (vg, rf)) in enumerate([(false, missing), (true, true), (true, false)])
		ax = plot_group_accuracy!(f_interleaved_grouped[1,i], 
			filter(x -> (x.valence_grouped .== vg) & (ismissing(rf) .|| x.reward_first == rf), PLT_data); 
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

# ╔═╡ 5d8615c6-0c8b-4d7a-bc4a-79df915aeb58
plot_split_by_condition(PLT_data)

# ╔═╡ 61cffe15-3e9e-447b-8fa7-2cde9a83d906
plot_split_by_condition(filter(x -> x.session == "1", PLT_data))

# ╔═╡ ca6b7a59-242e-44b1-9ef5-85759cfd9f93
plot_split_by_condition(filter(x -> x.session == "2", PLT_data))

# ╔═╡ Cell order:
# ╠═74c8335c-4095-11ef-21d3-0715bde378a8
# ╠═fb5e4cda-5cdd-492a-8ca2-38fc3fc68ce9
# ╠═1b7c9fc7-af54-4e2f-8303-b64ddd519453
# ╠═1af9e9df-0d5c-4f7b-a14f-9fc69a5e4d9e
# ╠═d97ba043-122c-47b8-ab3e-b3d157f47f42
# ╠═c6ce2aee-24d4-49f8-a57c-2b4e9a3ca022
# ╠═48d11871-5cd3-40f7-adb8-92db011a5d98
# ╠═44985a70-bd56-4e61-a187-a7911c773457
# ╠═bfd2018a-b45c-4eba-b898-39b2a5f10134
# ╠═3c0fa20c-5543-4c92-92d6-3d4495d2cdf5
# ╠═75162f83-a5f5-44f8-a62c-b0d318604074
# ╠═5d8615c6-0c8b-4d7a-bc4a-79df915aeb58
# ╠═61cffe15-3e9e-447b-8fa7-2cde9a83d906
# ╠═ca6b7a59-242e-44b1-9ef5-85759cfd9f93
