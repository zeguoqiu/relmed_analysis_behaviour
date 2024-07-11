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
end

# ╔═╡ b2f5b51e-360d-4a9c-bdb4-fd96b82c2919
begin
	# Define REDCap API URL and token
	api_url = "https://redcap.slms.ucl.ac.uk/api/"
	api_token = "44F85D7A59D210F27C5233D8B39849D9"
end

# ╔═╡ 20590ede-3ea0-4596-a04f-412c84a93311
function get_REDCap_file(
	record_id::Int64
)
	# Create the payload for getting the file
	file_payload = Dict(
		"token" => api_token,
	    "content" => "file",
		"action" => "export",
		"record" => "13",
		"field" => "other_data",
		"returnFormat" => "json"
	)

	# Make the POST request to the REDCap API
	file = HTTP.post(api_url, body=HTTP.Form(file_payload), verbose = true)

	# Parse
	return JSON.parse(String(file.body))
end

# ╔═╡ 0c7b2544-cc07-4219-b262-2af9dae2e320
function get_REDCap_data()

	# Get the records --------
	# Create the payload for getting the record details
	rec_payload = Dict(
		"token" => api_token,
	    "content" => "record",
	    "action" => "export",
	    "format" => "json",
	    "type" => "flat",
	    "csvDelimiter" => "",
	    "rawOrLabel" => "raw",
	    "rawOrLabelHeaders" => "raw",
	    "exportCheckboxLabel" => "false",
	    "exportSurveyFields" => "false",
	    "exportDataAccessGroups" => "false",
	    "returnFormat" => "json"
	)

	# Make the POST request to the REDCap API
	record = HTTP.post(api_url, body=HTTP.Form(rec_payload), verbose = true)

	# Parse the JSON response
	record = JSON.parse(String(record.body))

	# Get the files
	jspsych_data = []
	for r in record
		if r["other_data"] == "file"
			push!(jspsych_data, get_REDCap_file(parse(Int64, r["record_id"])))
		end
	end

	# Convert to DataFrame
	jspsych_data = reduce(vcat, jspsych_data)

	jspsych_data = vcat(
		[DataFrame(d) for d in jspsych_data]...,
		cols=:union
	)
	
	return jspsych_data
end

# ╔═╡ 1588fe86-cf3d-496a-8199-3abb2c36232a
data = get_REDCap_data()

# ╔═╡ d5369cbb-696b-4caf-8986-5ea4f983970a
function whitelist(data)
	whitelist = combine(groupby(data, [:prolific_pid, :session]),
		:trialphase => (x -> "experiment_end_message" in x) => :finished,
		:outcomes => 
			(x -> filter(y -> !ismissing(y), unique(data.outcomes))[1]) => :bonus,
		:trial_type => (x -> sum(x .== "PLT")) => :n_trial_PLT,
		:block => (x -> filter(y -> typeof(y) == Int64, x) |> unique |> length) => :n_blocks_PLT
	)
end

# ╔═╡ e81fd624-ad5a-4dda-80d6-b55681a177d7
whitelist(data)

# ╔═╡ Cell order:
# ╠═053887ce-3f7d-11ef-0ab9-53a5a4738d90
# ╠═b2f5b51e-360d-4a9c-bdb4-fd96b82c2919
# ╠═20590ede-3ea0-4596-a04f-412c84a93311
# ╠═0c7b2544-cc07-4219-b262-2af9dae2e320
# ╠═1588fe86-cf3d-496a-8199-3abb2c36232a
# ╠═d5369cbb-696b-4caf-8986-5ea4f983970a
# ╠═e81fd624-ad5a-4dda-80d6-b55681a177d7
