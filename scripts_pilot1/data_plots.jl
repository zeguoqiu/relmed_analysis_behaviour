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
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP, JLD2, Dates, RCall, LinearAlgebra

	include("fetch_preprocess_data.jl")
	include("plotting_functions.jl")
	include("stan_functions.jl")
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
	PLT_data = exclude_PLT_sessions(PLT_data)
	nothing
end

# ╔═╡ 95b7d6fa-b85f-4095-b53d-afb5fabe9095
md"""
# Data plots from first pilot of PLT
This notebook contains plot of raw data from the first PLT pilot.

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

	save("results/acc_curve_by_valence.pdf", 
		f_acc_valence, pt_per_unit = 1)


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

# ╔═╡ 0ac2f4bd-b64c-4437-b3aa-3d1f2938c3dd
begin
	no_early_data = filter(x -> !x.early_stop, PLT_data)

	no_early_data.confusing = ifelse.(
		no_early_data.optimalRight .== 1, 
		no_early_data.outcomeRight .< no_early_data.outcomeLeft,
		no_early_data.outcomeLeft .< no_early_data.outcomeRight
	)

	transform!(groupby(no_early_data, [:prolific_pid, :session, :block]),
		:confusing => (x -> string(findall(x))) => :confusing_sequence
	)

	sequences = unique(no_early_data.confusing_sequence)
	sequences = DataFrame(
		confusing_sequence = sequences,
		sequence = 1:length(sequences)
	)

	no_early_data = leftjoin(no_early_data, sequences, on = :confusing_sequence)

	function plot_by_sequence(
		data::DataFrame;
		group::Union{Symbol, Missing}=missing
	)
		by_sequence = groupby(data, :sequence)
	
		f_sequences = Figure(size = (600, 800))
		axs = []
		for (i, gdf) in enumerate(by_sequence)
		
			r = div(i - 1, 5) + 1
			c = rem(i - 1, 5) + 1
			
			ax = plot_group_accuracy!(
				f_sequences[r, c], 
				gdf; 
				group = group,
				linewidth = 1.
			)
	
			if !("Int64[]" in gdf.confusing_sequence)
				vlines!(
					ax,
					filter(x -> x <= 13, eval(Meta.parse(unique(gdf.confusing_sequence)[1])) .+ 1),
					color = :red,
					linewidth = 1
				)
			end

			#vlines!(ax, [6], linestyle = :dash, color = :grey)
			
			hideydecorations!(ax, ticks = c != 1, ticklabels = c != 1)
			hidexdecorations!(ax, 
				ticks = length(by_sequence) >= (5 * r + c), 
				ticklabels = length(by_sequence) >= (5 * r + c))
			hidespines!(ax)
	
			push!(axs, ax)
			
		end
	
		linkyaxes!(axs...)
		
		f_sequences
	end

	f_sequences = plot_by_sequence(no_early_data)

	save("results/acc_by_sequence.png", f_sequences, pt_per_unit = 1)

	f_sequences
end

# ╔═╡ 90f87386-c510-4586-8739-e89ff6e67dac
let
	avg_acc = combine(
		groupby(no_early_data, [:prolific_pid, :session, :block]),
		:isOptimal => mean => :overall_acc
	)

	avg_acc = combine(
		groupby(avg_acc, [:prolific_pid, :session]),
		:overall_acc => mean => :overall_acc
	)

	avg_acc.overall_acc_grp = avg_acc.overall_acc .> median(avg_acc.overall_acc)

	no_early_data = leftjoin(no_early_data, avg_acc, 
		on = [:prolific_pid, :session],
		order = :left
	)

	f_seuqnce_overall_acc = plot_by_sequence(
		no_early_data; 
		group = :overall_acc_grp
	)

	Legend(
		f_seuqnce_overall_acc[0, 1:5],
		[LineElement(color = c, linewidth = 3) for c in Makie.wong_colors()[[2, 1]]],
		["Best performing half", "Worst performing half"],
		framevisible = false,
		orientation = :horizontal
	)

	rowsize!(f_seuqnce_overall_acc.layout, 0, Relative(0.01))

	save("results/acc_by_sequence+overal_acc.png", 
		f_seuqnce_overall_acc, pt_per_unit = 1)

	f_seuqnce_overall_acc

end

# ╔═╡ 2d4123e5-b9fa-42f6-b7d3-79cfd7b96395
km0_draws = let
	forfit = transform(
		groupby(no_early_data, [:prolific_pid, :session, :block]),
		:confusing => (x -> circshift(x, 1) .+ 0) => :prev_confusing
	)

	filter!(x -> x.trial > 1, forfit)

	saved_model_fld = "saved_models/kernel_method"

	csv_file = joinpath(saved_model_fld, "kernel_method_data.csv")

	CSV.write(csv_file, forfit)

	model_file = joinpath(saved_model_fld, "km0.rda")

	draws_file = joinpath(saved_model_fld, "km0_draws.csv")


	rscript = """
		library(brms)
		library(cmdstanr)
	    set_cmdstan_path("/home/jovyan/.cmdstanr/cmdstan-2.34.1")

		dat <- read.csv("$csv_file")

		# Trial as factor
		dat\$trial <- factor(dat\$trial)
		contrasts(dat\$trial) <- contr.sum(length(unique(dat\$trial)))
	
		km0 <- brm(
			isOptimal ~ 1 + trial * prev_confusing + 
				(1 + trial * prev_confusing | prolific_pid),
			dat,
			family = bernoulli(),
			prior = prior(normal(0,1), class = "b") +
				prior(normal(0,1), class = "sd") +
				prior(lkj(2), class = "cor"),
			backend = "cmdstanr",
			threads = 3,
			cores = 4,
			file = "$model_file"
		)
		
		write.csv(as.data.frame(km0), file = "$draws_file")

		km0
	"""

	if !isfile(draws_file)
		fit_summary = RCall.reval(rscript)
	end

	km0_draws = DataFrame(CSV.File(draws_file))

end

# ╔═╡ b83502e2-53da-4808-86db-9f8b2add5caa
begin
	function plot_kernel_method(;
		draws::DataFrame,
		filter::Regex,
		ylabel::String = "Effect on subsequent accuracy",
		apply_contrasts::Bool = true
	)
		# Select only needed columns from draws
		confusing_trial_coefs = Matrix(select(draws, filter))

		if apply_contrasts
			# This is the contast matrix used to fit the data
			contrasts = vcat(Matrix(1.0I, 11, 11), fill(-1., (1, 11)))
		
			# Multiply contrast matrix by coefficients, and add the prev_confusing mean coefficient
			kernel_coefs = confusing_trial_coefs[:, 1] .+ transpose(contrasts * transpose(confusing_trial_coefs[:, 2:end]))
		else
			kernel_coefs = confusing_trial_coefs
		end
	
		# Compute summary statistics of posterior distribution
		kernel_coefs_m = median(kernel_coefs, dims = 1)
	
		kernel_coefs_lb = quantile.(eachcol(kernel_coefs), 0.25)
	
		kernel_coefs_ub = quantile.(eachcol(kernel_coefs), 0.75)
	
		kernel_coefs_llb = quantile.(eachcol(kernel_coefs), 0.025)
	
		kernel_coefs_uub = quantile.(eachcol(kernel_coefs), 0.975)
	
		# Plot
		f_kernel = Figure()
	
		ax = Axis(
			f_kernel[1,1],
			xlabel = "Confusing trial #",
			ylabel = ylabel,
			xticks = 1:length(kernel_coefs_m),
		)
	
		band!(
			ax,
			1:length(kernel_coefs_m),
			vec(kernel_coefs_llb),
			vec(kernel_coefs_uub),
			color = (Makie.wong_colors()[1], 0.1)
		)
	
		band!(
			ax,
			1:length(kernel_coefs_m),
			vec(kernel_coefs_lb),
			vec(kernel_coefs_ub),
			color = (Makie.wong_colors()[1], 0.3)
		)
	
		lines!(
			ax,
			1:length(kernel_coefs_m),
			vec(kernel_coefs_m)
		)
	
		hlines!([0.], linestyle = :dash, color = :grey)

		return f_kernel
	end

	f_kernel = plot_kernel_method(;
		draws = km0_draws,
		filter = r"b_.*confusing"
	)

	save("results/kernel_method_confusing_feedback.png", 
		f_kernel, pt_per_unit = 1)

	f_kernel

end

# ╔═╡ 344e9f00-c8a7-41b1-a2a2-5074e59044cc
let
	f_kernel_cor = plot_kernel_method(;
			draws = km0_draws,
			filter = r"cor_prolific_pid__Intercept__trial.*confusing",
			ylabel = "Correlation between intercept\nand confusing trial coefficient",
			apply_contrasts = false
		)
	
	save("results/kernel_method_confusing_feedback_corr.png", 
		f_kernel_cor, pt_per_unit = 1)

	f_kernel_cor
end

# ╔═╡ bacc8ce8-4481-4142-a582-f4fc673374b7
print_CI(x) = 
"$(round(median(x), digits = 2)), 95% CI=[$(round(llb(x), digits = 2)), $(round(uub(x), digits = 2))]"

# ╔═╡ 07a30c6b-8956-42f5-861e-6de7792b6c7b
km0_draws[!, Symbol("cor_prolific_pid__Intercept__prev_confusing")] |>
	print_CI

# ╔═╡ 954b4741-fb6c-4afd-bd5c-94722a01fc29
names(select(km0_draws, r"cor_prolific_pid__Intercept.*confusing"))

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
# ╠═0ac2f4bd-b64c-4437-b3aa-3d1f2938c3dd
# ╠═90f87386-c510-4586-8739-e89ff6e67dac
# ╠═2d4123e5-b9fa-42f6-b7d3-79cfd7b96395
# ╠═b83502e2-53da-4808-86db-9f8b2add5caa
# ╠═344e9f00-c8a7-41b1-a2a2-5074e59044cc
# ╠═bacc8ce8-4481-4142-a582-f4fc673374b7
# ╠═07a30c6b-8956-42f5-861e-6de7792b6c7b
# ╠═954b4741-fb6c-4afd-bd5c-94722a01fc29
