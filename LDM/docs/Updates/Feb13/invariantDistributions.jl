using Serialization, StatsPlots, StatsBase, Plots, Statistics;

datapath = "/home/cmhyett/Work/LDM/results/ReynoldsGeneralization/";
invars = Dict(240=>deserialize(datapath * "./invars_250.jls"),
              430=>deserialize(datapath * "./invars_450.jls"),
              610=>deserialize(datapath * "./invars_650.jls"));

for Re in keys(invars)
    ps = [];
    boxPlotInvars = zeros(size(invars[Re]));
    for i in 1:5
        arr = invars[Re][i,:];
        arr ./= mean(arr);
        push!(ps, histogram(arr, legend=false, title="\$ \\lambda_{$(i)}/\\langle \\lambda_{$(i)} \\rangle \$", xlims = (-10, 10), normalize = :pdf, yaxis=:log));
        boxPlotInvars[i,:] .= arr;
    end
    push!(ps, groupedboxplot(reshape(boxPlotInvars, (5, size(boxPlotInvars)[end], 1)), outliers=false, legend=false));
    
    p = plot(ps..., layout=(2,3), size=(1200,800), plot_title="\$Re \\approx $(Re) \$");
    savefig(p, "/home/cmhyett/Work/LDM/docs/Updates/Feb13/figs/zoomedInvars_$(Re).png");
end
