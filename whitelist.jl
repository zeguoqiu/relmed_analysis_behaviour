### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 053887ce-3f7d-11ef-0ab9-53a5a4738d90
begin
	import Pkg
	
	# activate the shared project environment
    Pkg.activate("relmed_environment")
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate
	using CairoMakie, Random, DataFrames, Distributions, Printf, PlutoUI, StatsBase, JSON, CSV, HTTP

	include("fetch_preprocess_data.jl")
end

# ╔═╡ d5369cbb-696b-4caf-8986-5ea4f983970a
function whitelist(data::DataFrame)
	whitelist = combine(groupby(data, [:prolific_pid, :record_id, :exp_start_time, :session, :condition]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:trialphase => (x -> "kick-out" in x) => :kick_out,
		:outcomes => 
			(x -> filter(y -> !ismissing(y), unique(x)) |> z -> length(z) > 0 ? z[1] : missing) => :bonus,
		[:trial_type, :block] => ((t, b) -> sum((t .== "PLT") .& (typeof.(b) .== Int64))) => :n_trial_PLT,
		:block => (x -> filter(y -> typeof(y) == Int64, x) |> unique |> length) => :n_blocks_PLT,
		:n_warnings => maximum => :n_warnings
	)

	sort!(whitelist, :n_blocks_PLT, rev = true)

	return whitelist
end

# ╔═╡ 0127aa42-9506-4c6f-8f43-dd4308b3195c
begin
	jspsych_data, records = get_REDCap_data()
end

# ╔═╡ 637edbe6-2e18-4129-b3ee-90aea2c41616
data = REDCap_data_to_df(jspsych_data, records)

# ╔═╡ e81fd624-ad5a-4dda-80d6-b55681a177d7
begin
	remove_testing!(data)
	nothing
end

# ╔═╡ d364b7ea-dac8-4b04-9e30-7cf55a383322
begin
	wl = whitelist(data)
	
	sort!(wl, [:session, :condition])
end

# ╔═╡ e465d30f-1f2a-402c-a279-5a18922c3085
let cond = "00",
	date = "2024-07-13",
	session = "2"

	twl = filter(x -> (x.condition == cond) & 
		occursin(date, x.exp_start_time) &
		(x.session == session), wl)

	println("didnt finish")
	bl = filter(x -> x.n_blocks_PLT < 24, twl)
	for r in eachrow(bl)
		println("$(r.prolific_pid)")
	end
	
	println("bonus")
	filter!(x -> x.n_blocks_PLT ==24, twl)
	for r in eachrow(twl)
		println("$(r.prolific_pid),$(r.bonus)")
	end

	println("\nApprove")
	for r in eachrow(unique(twl[!, :prolific_pid]))
		println(r[1])
	end


	println("\nDouble takers")
	doubles = filter(x -> x.n > 1, 
		combine(groupby(twl, :prolific_pid), :bonus => length => :n))

	for_second_sess = filter(x -> !(x.prolific_pid in doubles.prolific_pid), twl)
	println("\nSecond session, n=$(nrow(for_second_sess))")
	for r in eachrow(for_second_sess)
		println("$(r.prolific_pid)")
	end
	

	
end

# ╔═╡ 998dcced-2cfe-41f8-9fd1-ad0fc9ee2eea
PLT_data = prepare_PLT_data(data)

# ╔═╡ Cell order:
# ╠═053887ce-3f7d-11ef-0ab9-53a5a4738d90
# ╠═d5369cbb-696b-4caf-8986-5ea4f983970a
# ╠═0127aa42-9506-4c6f-8f43-dd4308b3195c
# ╠═637edbe6-2e18-4129-b3ee-90aea2c41616
# ╠═e81fd624-ad5a-4dda-80d6-b55681a177d7
# ╠═d364b7ea-dac8-4b04-9e30-7cf55a383322
# ╠═e465d30f-1f2a-402c-a279-5a18922c3085
# ╠═998dcced-2cfe-41f8-9fd1-ad0fc9ee2eea
