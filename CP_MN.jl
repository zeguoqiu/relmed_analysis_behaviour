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
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP

	include("fetch_preprocess_data.jl")
end

# ╔═╡ 3deb5d2c-4094-11ef-2ff7-6587b949c7c4
function plot_group_accuracy!(
	f::GridPosition,
	data::DataFrame;
	group::Union{Symbol, Missing}=missing,
	colors = Makie.wong_colors(),
	backgroundcolor = :white
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
	ax = Axis(f,
		xlabel = "Trial #",
		ylabel = "Prop. optimal chioce",
		xautolimitmargin = (0., 0.),
		xticks = range(1, round(Int64, maximum(sum_data.trial)), 4),
		backgroundcolor = backgroundcolor
	)

	for (i,g) in enumerate(unique(sum_data.group))
		gdat = filter(:group => (x -> x==g), sum_data)
		println(i)
		println(g)
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

	return sum_data

end

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
# ╠═3deb5d2c-4094-11ef-2ff7-6587b949c7c4
