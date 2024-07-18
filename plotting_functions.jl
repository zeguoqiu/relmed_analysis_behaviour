# Functions to assist plotting

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