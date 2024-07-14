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

# ╔═╡ 09f9af53-2116-41db-8813-c8da1a7a4576
describe(PLT_data)

# ╔═╡ 78549c5d-48b4-4634-b380-b2b8d883d430
begin
	group_QLrs_sum, group_QLrs_draws, group_QLrs_time = load_run_cmdstanr(
		"group_QLrs",
		"group_QLrs.stan",
		to_standata(sim_dat,
			feedback_magnitudes,
			feedback_ns;
			model_name = "group_QLrs");
		print_vars = ["mu_a", "sigma_a", "mu_rho", "sigma_rho"],
		threads_per_chain = 3
	)
	group_QLrs_sum, group_QLrs_time
end

# ╔═╡ Cell order:
# ╠═e01188c3-ca30-4a7c-9101-987752139a71
# ╠═bdeadcc4-1a5f-4c39-a055-e61b3db3f3b1
# ╠═963e5f75-00f9-4fcc-90b9-7ecfb7e278f2
# ╠═09f9af53-2116-41db-8813-c8da1a7a4576
# ╠═78549c5d-48b4-4634-b380-b2b8d883d430
