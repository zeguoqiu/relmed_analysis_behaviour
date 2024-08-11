# Functions for plotting data and simulations

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