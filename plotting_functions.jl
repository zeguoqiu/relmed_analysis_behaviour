# Functions to assist plotting

# General helper functions ---------------------------------

# Convert a to α and back
	a2α(x) = cdf(Normal(), x)

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

# Raw behaviour ----------------------------------------------------
# Plot accuracy for a group, divided by condition / group
function plot_group_accuracy!(
    f::GridPosition,
    data::DataFrame;
    group::Union{Symbol, Missing} = missing,
    colors = Makie.wong_colors(),
    title::String = "",
    legend::Union{Dict, Missing} = missing,
    legend_title::String = "",
    backgroundcolor = :white,
    ylabel::Union{String, Makie.RichText}="Prop. optimal choice",
    levels::Union{AbstractVector, Missing} = missing
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

    group_levels = ismissing(levels) ? unique(sum_data.group) : levels
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

function plot_group_q_values(
	f::GridPosition,
	data::DataFrame;
	traces = true,
	legend = true,
	colors = Makie.wong_colors(),
	backgroundcolor = :white
	)

	p_data = copy(data)

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
			linewidth = .2)

		series!(ax, transpose(Matrix(gp_dat_B[!, 3:end])), 
			solid_color = (colors[g], 0.1),
			linewidth = .2,
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

function plot_q_value_acc!(
	f::Figure,
	sim_dat::DataFrame;
	legend = true,
	colors = Makie.wong_colors(),
	backgroundcolor = :white
	)
	
	plot_group_q_values(f[1 + legend ,1], sim_dat; 
		legend = legend, colors = colors, backgroundcolor = backgroundcolor)
	plot_group_accuracy(f[1 + legend,2], sim_dat;
		colors = colors, backgroundcolor = backgroundcolor)

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

## Fisher Information --------------------------------------------------|
# Plot Fisher Information as a function of block and trial #
function plot_blocks_trials!(f::GridPosition,
	FI::Matrix{Matrix{Float64}}, # Will be transposed for plotting 
	x::AbstractVector, 
	y::AbstractVector;
	xlabel::Vector{String},
	ylabel::String)

		min_FI = minimum(map(mat -> minimum(mat), FI))
		max_FI = maximum(map(mat -> maximum(mat), FI))


		axs = []
		for j in 1:size(FI,1)
			for i in 1:size(FI,2)
				ax = Axis(f[i,j],
					xlabel = i == 2 ? xlabel[j] : "",
					ylabel = j == 1 ? "$(["Blocks", "Trials"][i])\n$ylabel" : ""
				)

				push!(axs, ax)
	
				contour!(ax,
					x,
					y,
					FI[j,i],
					colorrange = (min_FI, max_FI)
				)
			end
		end

		linkaxes!(axs...)
end

# Average over dimension and squeeze the matrix
avg_over(m::AbstractArray; dims::Tuple{Vararg{Int64}}) = dropdims(mean(m, dims = dims); dims = dims)

# Convert a to α and back
a2α(x) = cdf(Normal(), x)
α2a(x) = quantile(Normal(), x)

# Average squares across the diagonal of a matrix
function average_diagonal_squares(matrix::Matrix{T}, n::Int) where T
    # Get the size of the matrix
    rows, cols = size(matrix)
    
    # Ensure the matrix is square and the block size fits
    if rows != cols
        throw(ArgumentError("The matrix must be square"))
    end
    
    if rows % n != 0
        throw(ArgumentError("The block size must evenly divide the dimensions of the matrix"))
    end

    # Number of blocks along the diagonal
    num_blocks = div(rows, n)
    
    # Array to store the stats
    med = Vector{Float64}(undef, num_blocks)
	lb = Vector{Float64}(undef, num_blocks)
	ub = Vector{Float64}(undef, num_blocks)
    
    # Loop through each block
    for i in 1:num_blocks
        # Define the indices for the current n x n block
        row_start = (i - 1) * n + 1
        row_end = i * n
        col_start = (i - 1) * n + 1
        col_end = i * n
        
        # Extract the n x n block
        block = matrix[row_start:row_end, col_start:col_end]
        
        # Calculate the average of the elements in the block
        med[i] = median(block)
		lb[i] = quantile(vec(block), 0.25)
		ub[i] = quantile(vec(block), 0.75)
    end
    
    return (med = med, lb = lb, ub = ub)
end

# Draw a sqaure on a plot
function draw_square!(i, j, δx, δy; color = :black, linewidth = 0.5)
    # Define the corners of the square
    corners = [
        (i - δx/2, j - δy/2), # Bottom-left
        (i - δx/2, j + δy/2), # Top-left
        (i + δx/2, j + δy/2), # Top-right
        (i + δx/2, j - δy/2), # Bottom-right
        (i - δx/2, j - δy/2)  # Back to Bottom-left to close the square
    ]
    lines!(corners, color = color, linewidth = linewidth)
end

## Posteriors --------------------------------------------------|
# Plot single-participant parameter values from a hierarchical model
function plot_p_params!(
	f::GridPosition,
	draws::DataFrame,
	sim_dat::DataFrame,
	param::String;
	param_label::String = param,
	ylabel::Union{AbstractString, Makie.RichText} = "Participant #"
	)

	# Select columns in draws DataFrame
	tdraws = copy(select(draws, Regex("$(param)\\[\\d+\\]")))

	# Add mean and multiply by SD
	tdraws .*= draws[!, Symbol("sigma_$param")]
	tdraws .+= draws[!, Symbol("mu_$param")]	
	
	# Stack
	tdraws = stack(tdraws)	

	# Summarise
	tdraws = combine(groupby(tdraws, :variable),
		:value => median => :median,
		:value => lb => :lb,
		:value => ub => :ub,
		:value => llb => :llb,
		:value => uub => :uub)
	

	# Create :PID columns
	get_PID(s) = parse(Int64, 
		replace(s, Regex("$param\\[(\\d+)\\]") => s"\1"))
	transform!(tdraws,
		:variable => ByRow(get_PID) => :PID)
	
	# Merge with true values
	true_params = unique(select(sim_dat, [:PID, Symbol(param)]))
	tdraws = innerjoin(tdraws, true_params, on = :PID)

	# Sort by median posterior
	sort!(tdraws, :median)

	# Make scatter plot
	ax_scatter = Axis(f[1,2],
		xlabel = "True $param_label",
		ylabel = "Posterior $param_label")

	scatter!(
		ax_scatter,
		tdraws[!, Symbol(param)],
		tdraws.median,
		markersize = 6
		)

	ablines!(
		ax_scatter,
		0.,
		1.,
		linestyle = :dash,
		color = :grey
	)

	# # Make snake plot
	ax_snake = Axis(f[1,1],
		xlabel = param_label,
		ylabel = ylabel
	)

	rangebars!(
		ax_snake,
		1:nrow(tdraws),
		tdraws.llb,
		tdraws.uub,
		direction = :x,
		color = (Makie.wong_colors()[1], 0.3)
	)


	rangebars!(
		ax_snake,
		1:nrow(tdraws),
		tdraws.lb,
		tdraws.ub,
		direction = :x,
		color = (Makie.wong_colors()[1], 0.5)
	)

	scatter!(
		ax_snake,
		tdraws.median,
		1:nrow(tdraws)
	)

	scatter!(
		ax_snake,
		tdraws[!, Symbol(param)],
		1:nrow(tdraws),
		color = :red,
		markersize = 6
	)

	return ax_snake, ax_scatter

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

# Plot prior predictive checks
function plot_prior_predictive(
	sum_fits::DataFrame;
	params::Vector{String} = ["a", "rho"],
	labels::AbstractVector = ["a", "ρ"],
	show_n::AbstractVector = unique(sum_fits.n_blocks), # Levels to show
	ms::Int64 = 7 # Marker size
)
		block_levels = unique(sum_fits.n_blocks)
		tsum_fits = copy(sum_fits)
		tsum_fits.colors = (x -> Dict(block_levels .=> 
			Makie.wong_colors()[1:length(block_levels)])[x]).(sum_fits.n_blocks)

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
				sum_fits[!, Symbol("$(p)_ub")],
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
    