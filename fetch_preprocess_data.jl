# Functions for fetching data and preprocessing it

begin
	# Define REDCap API URL and token
	api_url = "https://redcap.slms.ucl.ac.uk/api/"
	api_token = "44F85D7A59D210F27C5233D8B39849D9"
end

# Fetch one data file by record_id
function get_REDCap_file(
	record_id::String
)
	# Create the payload for getting the file
	file_payload = Dict(
		"token" => api_token,
	    "content" => "file",
		"action" => "export",
		"record" => record_id,
		"field" => "other_data",
		"returnFormat" => "json"
	)

	# Make the POST request to the REDCap API
	file = HTTP.post(api_url, body=HTTP.Form(file_payload), verbose = true)

	# Parse
	return JSON.parse(String(file.body))
end

# Fetch entire dataset
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
			tdata = get_REDCap_file(r["record_id"])

			# Add record_id
			for tr in tdata
				tr["record_id"] = r["record_id"]
			end
			
			push!(jspsych_data, tdata)
		end
	end

	
	return jspsych_data, record
end

# Convert to df and merge REDCap record data and jsPsych data
function REDCap_data_to_df(jspsych_data, records)
	
	# Records to df
	records_df = DataFrame(records)
	
	# Convert to DataFrame
	jspsych_data = reduce(vcat, jspsych_data)

	jspsych_data = vcat(
		[DataFrame(d) for d in jspsych_data]...,
		cols=:union
	)

	jspsych_data = leftjoin(jspsych_data, 
		rename(records_df[!, [:prolific_pid, :record_id, :start_time]],
			:start_time => :exp_start_time),
		on = [:prolific_pid, :record_id]
	)

	return jspsych_data
end

remove_testing!(data::DataFrame) = filter!(x -> !occursin(r"yaniv|tore", x.prolific_pid), data)

# Filter PLT data
function prepare_PLT_data(data::DataFrame)

	# Select rows
	PLT_data = filter(x -> x.trial_type == "PLT", data)

	# Select columns
	PLT_data = PLT_data[:, Not(map(col -> all(ismissing, col), eachcol(PLT_data)))]

	# Filter practice
	filter!(x -> typeof(x.block) == Int64, PLT_data)

	# Sort
	sort!(PLT_data, [:prolific_pid, :session, :block, :trial])

	return PLT_data

end