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

# ╔═╡ 0c7b2544-cc07-4219-b262-2af9dae2e320
function read_REDCap()

	# Define REDCap API URL and token
	api_url = "https://redcap.slms.ucl.ac.uk/api/"
	api_token = "44F85D7A59D210F27C5233D8B39849D9"

	# Create the payload for the POST request
	payload = Dict(
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

	HTTP.Form(payload)
	# Make the POST request to the REDCap API
	response = HTTP.post(api_url, body=HTTP.Form(payload))
	
	# Parse the JSON response
	data = JSON.parse(String(response.body))

	return data
end

# ╔═╡ 47b7379d-c0f7-4276-8904-8a17e01ee9e9
data = read_REDCap()

# ╔═╡ Cell order:
# ╠═053887ce-3f7d-11ef-0ab9-53a5a4738d90
# ╠═0c7b2544-cc07-4219-b262-2af9dae2e320
# ╠═47b7379d-c0f7-4276-8904-8a17e01ee9e9
