# Functions for plotting data and simulations

# Plot unit line
unit_line!(ax; color = :grey, linestyle = :dash, linewidth = 2) = ablines!(
	0., 
	1.,
	color = color,
	linestyle = linestyle,
	linewidth = linewidth
	)

# Regression line
function regression_line_func(df::DataFrame, 
    x::Symbol, 
    y::Symbol)
    
    # Compute the coefficients of the regression line y = mx + c
    X = hcat(ones(length(df[!, x])), df[!, x])  # Design matrix with a column of ones for the intercept
    β = X \ df[!, y]                    # Solve for coefficients (m, c)
    
    y_line(x_plot) = β[1] .+ β[2] * x_plot

    return y_line
end

range_regression_line(x::Vector{Float64}; res = 200) = 
    range(minimum(x), maximum(x), res)


# Plot scatter with lm regression line
function scatter_regression_line!(
	f::GridPosition,
	df::DataFrame,
	x_col::Symbol,
	y_col::Symbol,
	xlabel::String,
	ylabel::String;
	transform_x::Function = x -> x,
	transform_y::Function = x -> x,
	color = Makie.wong_colors()[1],
	legend::Union{Dict, Missing} = missing,
	legend_title::String = "",
	write_cor::Bool = true,
	cor_correction::Function = x -> x, # Correction to apply for correlation, e.g. Spearman Brown
	cor_label::String = "r",
	aspect::Float64 = 1.
)

	x = df[!, x_col]
	y = df[!, y_col]
	
	ax = Axis(f,
		xlabel = xlabel,
		ylabel = ylabel,
		subtitle = write_cor ? "$cor_label=$(round(
			cor_correction(cor(x, y)), digits= 2))" : "",
		aspect = aspect
	)

	# Regression line
	treg = regression_line_func(df, x_col, y_col)
	lines!(
		ax,
		range_regression_line(x) |> transform_x,
		treg.(range_regression_line(x)) |> transform_y,
		color = :grey,
		linewidth = 4
	)

	sc = scatter!(
		ax,
		transform_x.(x),
		transform_y.(y),
		markersize = 6,
		color = color
	)

	if !ismissing(legend)
		Legend(
			f,
			[MarkerElement(color = k, marker = :circle) for k in keys(legend)],
			[legend[k] for k in keys(legend)],
			legend_title,
			halign = :right,
			valign = :top,
			framevisible = false,
			tellwidth = false,
			tellheight = false
		)

	end
end


# Plot accuracy for a group, divided by condition / group
function plot_group_accuracy!(
    f::GridPosition,
    data::Union{DataFrame, SubDataFrame};
    group::Union{Symbol, Missing} = missing,
    pid_col::Symbol = :prolific_pid,
    acc_col::Symbol = :isOptimal,
    colors = Makie.wong_colors(),
    title::String = "",
    legend::Union{Dict, Missing} = missing,
    legend_title::String = "",
    backgroundcolor = :white,
    ylabel::Union{String, Makie.RichText}="Prop. optimal choice",
    levels::Union{AbstractVector, Missing} = missing,
	error_band::Union{Bool, String} = "se",
	linewidth::Float64 = 3.,
	plw::Float64 = 1.
    )

	# Set up axis
	ax = Axis(f[1,1],
        xlabel = "Trial #",
        ylabel = ylabel,
        xautolimitmargin = (0., 0.),
        backgroundcolor = backgroundcolor,
        title = title
    )

	plot_group_accuracy!(
		ax,
		data;
		group = group,
		pid_col = pid_col,
		acc_col = acc_col,
		colors = colors,
		title = title,
		legend = legend,
		legend_title = legend_title,
		backgroundcolor = backgroundcolor,
		ylabel = ylabel,
		levels = levels,
		error_band = error_band,
		linewidth = linewidth,
		plw = plw
		)

	# Legend
	if !ismissing(legend)
		group_levels = ismissing(levels) ? unique(data[!, group]) : levels
		elements = [PolyElement(color = colors[i]) for i in 1:length(group_levels)]
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

function plot_group_accuracy!(
    ax::Axis,
    data::Union{DataFrame, SubDataFrame};
    group::Union{Symbol, Missing} = missing,
    pid_col::Symbol = :prolific_pid,
    acc_col::Symbol = :isOptimal,
    colors = Makie.wong_colors(),
    title::String = "",
    legend::Union{Dict, Missing} = missing,
    legend_title::String = "",
    backgroundcolor = :white,
    ylabel::Union{String, Makie.RichText}="Prop. optimal choice",
    levels::Union{AbstractVector, Missing} = missing,
	error_band::Union{String, Bool} = "se", # Whether and which type of error band to plot
	linewidth::Float64 = 3.,
	plw::Float64 = 1. # Line width for per participant traces
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
        groupby(tdata, [pid_col, :group, :trial]),
        acc_col => mean => :acc
    )

	# Unstack per participant data for Stim A
	p_data = unstack(sum_data, [:group, :trial], 
		pid_col,
		:acc)

    sum_data = combine(
        groupby(sum_data, [:group, :trial]),
        :acc => mean => :acc,
        :acc => sem => :acc_sem,
		:acc => lb => :acc_lb,
		:acc => ub => :acc_ub,
		:acc => llb => :acc_llb,
		:acc => uub => :acc_uub
    )

	# Set axis xticks
	ax.xticks = range(1, round(Int64, maximum(sum_data.trial)), 4)

    group_levels = ismissing(levels) ? unique(sum_data.group) : levels
    for (i,g) in enumerate(group_levels)
        gdat = filter(:group => (x -> x==g), sum_data)
		g_p_dat = filter(:group => (x -> x == g), p_data)

        # Plot line
		mc = length(colors)

		if typeof(error_band) == String
			if error_band == "PI"
				band!(ax,
					gdat.trial,
					gdat.acc_llb,
					gdat.acc_uub,
					color = (colors[rem(i - 1, mc) + 1], 0.1)
				)
			end 

			if error_band in ["se", "PI"]
				band!(ax,
					gdat.trial,
					error_band == "se" ? gdat.acc - gdat.acc_sem : gdat.acc_lb,
					error_band == "se" ? gdat.acc + gdat.acc_sem : gdat.acc_ub,
					color = (colors[rem(i - 1, mc) + 1], 0.3)
				)
			elseif error_band == "traces"
				series!(ax, transpose(Matrix(g_p_dat[!, 3:end])), 
					solid_color = (colors[rem(i - 1, mc) + 1], 0.1),
					linewidth = plw)
			end
		end
        
        lines!(ax, 
            gdat.trial, 
            gdat.acc, 
            color = colors[rem(i - 1, mc) + 1],
            linewidth = linewidth)
    end
        
end

function plot_sim_group_q_values!(
	f::GridPosition,
	data::DataFrame;
	traces = true,
	legend = true,
	colors = Makie.wong_colors(),
	backgroundcolor = :white,
	plw = 0.2
	)

	p_data = copy(data)

	# Normalize by ρ
	p_data.EV_A_s = p_data.EV_A ./ p_data.ρ
	p_data.EV_B_s = p_data.EV_B ./ p_data.ρ

	# Summarize into proportion of participants choosing optimal
	p_data = combine(
		groupby(p_data, [:group, :trial, :PID]),
		:EV_A_s => mean => :EV_A,
		:EV_B_s => mean => :EV_B
	)

	# Unstack per participant data for Stim A
	p_data_A = unstack(p_data, [:group, :trial], 
		:PID,
		:EV_A)

	# Unstack per participant data for Stim B
	p_data_B = unstack(p_data, [:group, :trial], 
		:PID,
		:EV_B)

	# Summarize per group
	sum_data = combine(
		groupby(p_data, [:group, :trial]),
		:EV_A => mean => :EV_A,
		:EV_B => mean => :EV_B
	)


	# Set up axis
	ax = Axis(f,
		xlabel = "Trial #",
		ylabel = "Q value",
		xautolimitmargin = (0., 0.),
		yautolimitmargin = (0., 0.),
		xticks = range(1, round(Int64, maximum(sum_data.trial)), 4),
		backgroundcolor = backgroundcolor)

	# Plot line
	for g in unique(sum_data.group)

		# Subset per group
		gsum_dat = filter(:group => (x -> x == g), sum_data)
		gp_dat_A = filter(:group => (x -> x == g), p_data_A)
		gp_dat_B = filter(:group => (x -> x == g), p_data_B)


		# Plot group means
		lines!(ax, gsum_dat.trial, gsum_dat.EV_A, 
			color = colors[g],
			linewidth = 3)
		lines!(ax, gsum_dat.trial, gsum_dat.EV_B, 
			color = colors[g],
			linestyle = :dash,
			linewidth = 3)

		# Plot per participant
		series!(ax, transpose(Matrix(gp_dat_A[!, 3:end])), 
			solid_color = (colors[g], 0.1),
			linewidth = plw)

		series!(ax, transpose(Matrix(gp_dat_B[!, 3:end])), 
			solid_color = (colors[g], 0.1),
			linewidth = plw,
			linestyle = :dash)

		if legend
			# Add legend for solid and dashed lines
			Legend(f,
				[[LineElement(color = :black)],
				[LineElement(color = :black,
					linestyle = :dash)]],
				[["Stimulus A"],
				["Stimulus B"]],
				tellheight = false,
				tellwidth = false,
				halign = :left,
				valign = :top,
				framevisible = false,
				nbanks = 2)
		end

	end
end

function plot_sim_q_value_acc!(
	f::Union{Figure, GridLayout},
	sim_dat::DataFrame;
	legend = true,
	colors = Makie.wong_colors(),
	backgroundcolor = :white,
	plw = 0.2,
	acc_error_band = "se"
	)

    # Calcualte accuracy
    sim_dat.isOptimal = sim_dat.choice .== 1
	
	plot_sim_group_q_values!(f[1 + legend ,1], sim_dat; 
		legend = legend, colors = colors, backgroundcolor = backgroundcolor, plw = plw)
	plot_group_accuracy!(f[1 + legend,2], sim_dat;
        group = :group, pid_col = :PID,
		colors = colors, backgroundcolor = backgroundcolor, error_band = acc_error_band)

	if (legend)
		# Add legend
		group_params = combine(groupby(sim_dat, :group), # Get empirical group parameters
			:α => mean => :α_mean,
			:α => std => :α_sd,
			:ρ => mean => :ρ_mean,
			:ρ => std => :ρ_sd,
			)

		Legend(f[1,1:2],
			[[LineElement(color = Makie.wong_colors()[g])] for g in 1:3],
			["$(n2l(g)): α=$(@sprintf("%0.2f", 
				group_params[g,:α_mean]))±$(@sprintf("%0.2f", 
				group_params[g,:α_sd])), ρ=$(@sprintf("%0.2f", 
				group_params[g,:ρ_mean]))±$(@sprintf("%0.2f", 
				group_params[g,:ρ_sd]))" for g in 1:3],
			"Group",
			framevisible = false,
			nbanks = 3,
			tellheight = false,
			titleposition = :left
		)

		rowsize!(f.layout, 1, Relative(0.05))
		rowgap!(f.layout, 10)	
	end

	return f
end

# This function makes density plots for posteriors, plus true value if needed
function plot_posteriors(draws::Vector{DataFrame},
	params::Vector{String};
	labels::AbstractVector = params,
	true_values::Union{Vector{Float64}, Nothing} = nothing,
	colors::AbstractVector = Makie.wong_colors()[1:length(draws)],
	nrows::Int64=1,
	scale_col::Union{Symbol, Nothing} = nothing,
	mean_col::Union{Symbol, Nothing} = nothing,
	model_labels::AbstractVector = repeat([nothing], length(draws))
)

	# Plot
	f_sim = Figure(size = (700, 20 + 230 * nrows))

	n_per_row = ceil(Int64, length(params) / nrows)

	for (i, p) in enumerate(params)

		# Set up axis
		ax = Axis(
			f_sim[div(i-1, n_per_row) + 1, rem(i - 1, n_per_row) + 1],
			xlabel = labels[i]
		)

		hideydecorations!(ax)
		hidespines!(ax, :l)

		for (j, d) in enumerate(draws)

			# Scale and add mean
			dp = copy(d[!, Symbol(p)])

			if !isnothing(scale_col)
				dp .*= d[!, scale_col]
			end

			if !isnothing(mean_col)
				dp .+= d[!, mean_col]
			end

			# Plot posterior density
			density!(ax,
				dp,
				color = (:black, 0.),
				strokewidth = 2,
				strokecolor = colors[j]
			)

			# Plot 95% PI
			linesegments!(ax,
				[(quantile(dp, 0.025), 0.),
				(quantile(dp, 0.975), 0.)],
				linewidth = 4,
				color = colors[j]
			)
		end

		# Plot true value
		if !isnothing(true_values)
			vlines!(ax,
				true_values[i],
				color = :gray,
				linestyle = :dash)
		end

	end

	draw_legend = length(draws) > 1 & !all(isnothing.(model_labels))
	if draw_legend
		Legend(f_sim[0, 1:n_per_row],
			[LineElement(color = Makie.wong_colors()[i]) for i in 1:length(draws)],
			model_labels,
			nbanks = length(draws),
			framevisible = false,
			valign = :bottom
		)
	end

	Label(f_sim[0, 1:n_per_row],
		rich(rich("Posterior distributions\n", fontsize = 18),
			rich("95% PI and true value marked as dashed line", fontsize = 14)),
		valign = :top
		)

	rowsize!(f_sim.layout, 0, Relative(draw_legend ? 0.4 / nrows : 0.2 / nrows))

	return f_sim
end

# Methods for MCMCChains
function plot_posteriors(draws::AbstractVector,
	params::Vector{String};
	labels::AbstractVector = params,
	true_values::Union{Vector{Float64}, Nothing} = nothing,
	colors::AbstractVector = Makie.wong_colors()[1:length(draws)],
	nrows::Int64=1,
	scale_col::Union{Symbol, Nothing} = nothing,
	mean_col::Union{Symbol, Nothing} = nothing,
	model_labels::AbstractVector = repeat([nothing], length(draws))
)	

	

	draws_dfs = [
		DataFrame(Array(chains), names(chains, :parameters)) for chains in draws
	]

	plot_posteriors(draws_dfs,
		params;
		labels = labels,
		true_values = true_values,
		colors = colors,
		nrows = nrows,
		scale_col = scale_col,
		mean_col = mean_col,
		model_labels = model_labels
	)	

end

# Plot prior predictive checks
function plot_SBC(
	sum_fits::DataFrame;
	params::Vector{String} = ["a", "rho"],
	labels::AbstractVector = ["a", "ρ"],
	show_n::AbstractVector = unique(sum_fits.n_blocks), # Levels to show
	ms::Int64 = 7 # Marker size
)		
	tsum_fits = copy(sum_fits)
	
	if !("n_blocks" in names(tsum_fits))
		tsum_fits[!, :n_blocks] .= 1
	end

	block_levels = unique(tsum_fits.n_blocks)
	
	tsum_fits.colors = (x -> Dict(block_levels .=> 
		Makie.wong_colors()[1:length(block_levels)])[x]).(tsum_fits.n_blocks)

	# Plot for each parameter
	f_sims = Figure(size = (700, 50 + 200 * length(params)))

	axs = []

	tsum_fits = filter(x -> x.n_blocks in show_n, tsum_fits)

	for (i, p) in enumerate(params)
		# Plot posterior value against real value
		ax = Axis(f_sims[i, 1],
			xlabel = rich("True ", labels[i]),
			ylabel = rich("Posterior estimate of ", labels[i])
			)

		rangebars!(ax,
			tsum_fits[!, Symbol("true_$(p)")],
			tsum_fits[!, Symbol("$(p)_lb")],
			tsum_fits[!, Symbol("$(p)_ub")],
			color = tsum_fits.colors
		)

		scatter!(ax,
			tsum_fits[!, Symbol("true_$(p)")],
			tsum_fits[!, Symbol("$(p)_m")],
			color = tsum_fits.colors,
			markersize = ms
		)

		ablines!(ax,
			0.,
			1.,
			linestyle = :dash,
			color = :gray,
			linewidth = 1)

		# Plot residual against real value						
		ax = Axis(f_sims[i, 2],
			xlabel = rich("True ", labels[i]),
			ylabel = rich("Posterior", labels[i], " - true ",  labels[i])
			)

		scatter!(ax,
			tsum_fits[!, Symbol("true_$(p)")],
			tsum_fits[!, Symbol("$(p)_sm")],
			color = tsum_fits.colors,
			markersize = ms
		)

		hlines!(ax, 0., linestyle = :dash, color = :grey)

		# Plot contraction against real value
		ax = Axis(f_sims[i, 3],
			xlabel = rich("True ", labels[i]),
			ylabel = rich("Posterior contraction of ", labels[i])
			)

		scatter!(ax,
			tsum_fits[!, Symbol("true_$(p)")],
			tsum_fits[!, Symbol("$(p)_cntrct")],
			color = tsum_fits.colors,
			markersize = ms
		)

	end

	if length(block_levels) > 1
		Legend(f_sims[0,1:3], 
			[MarkerElement(color = Makie.wong_colors()[i], marker = :circle) for i in 1:length(block_levels)],
			["$n" for n in block_levels],
			"# of blocks",
			nbanks = length(block_levels),
			framevisible = false,
			titleposition = :left)

		rowsize!(f_sims.layout, 0, Relative(0.05))
	end
	
	return f_sims
end