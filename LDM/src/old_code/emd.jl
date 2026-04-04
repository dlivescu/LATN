using Ipopt, JuMP, LinearAlgebra, StatsBase;
using HiGHS;

function setup()
    xs = -50.0:0.5:50.0;
    mids = midpoints(xs);

    d1 = rand(length(mids));
    d2 = zeros(length(mids));
    for i in 1:length(mids)
        d2[i] = i*rand();
    end
    d1 /= sum(d1);
    d2 /= sum(d2);

    dist = zeros(length(mids), length(mids));
    for i in 1:length(mids)
        for j in 1:length(mids)
            dist[i,j] = norm(mids[i] - mids[j]);
        end
    end
    return d1, d2, dist;
end

function setup2d()
    xs = -1.0:0.05:1.0;
    ys = -1.0:0.05:1.0;
    xmids = midpoints(xs);
    ymids = midpoints(ys);

    d1 = rand(length(xmids), length(ymids));
    d2 = rand(length(xmids), length(ymids));
    for i in 1:length(xmids)
        for j in 1:length(ymids)
            d2[i,j] = i*j*rand();
        end
    end
    d1 /= sum(d1);
    d2 /= sum(d2);

    dist = zeros(length(xmids), length(ymids), length(xmids), length(ymids));
    for i in 1:length(xmids)
        for j in 1:length(ymids)
            for k in 1:length(xmids)
                for l in 1:length(ymids)
                    dist[i,j,k,l] = norm([xmids[i] ymids[j]] - [xmids[k] ymids[l]]);
                end
            end
        end
    end
    d1 = reshape(d1, prod(size(d1)));
    d2 = reshape(d2, prod(size(d2)))
    dist = reshape(dist, (prod(size(dist)[1:2]), prod(size(dist)[3:4])));
    return d1, d2, dist
end

function locate_feasible_point(d1, d2, dist)
    wi = maximum(dist, dims=2)
    yj = maximum(dist, dims=1)
    x_guess = zeros(length(d1), length(d2));
    supply = copy(d1);
    demand = copy(d2);
    tab = zeros(length(d1), length(d2));
    for i in 1:size(tab)[1]
        for j in 1:size(tab)[2]
            tab[i,j] = wi[i] + yj[j] - dist[i,j]
        end
    end

    while (maximum(tab) > 0) #demand not satisfied and supply not exhausted
        supply_ind, demand_ind = Tuple(argmax(tab));
        if (argmin([supply[supply_ind], demand[demand_ind]]) == 1) #supply < demand
            x_guess[supply_ind, demand_ind] = supply[supply_ind]
            tab[supply_ind,:] .= -Inf;
        else
            x_guess[supply_ind, demand_ind] = demand[demand_ind];
            tab[:,demand_ind] .= -Inf;
        end
        demand[demand_ind] -= x_guess[supply_ind, demand_ind]
        supply[supply_ind] -= x_guess[supply_ind, demand_ind]
    end 
    return x_guess
end

function emd(d1, d2, dist; epsilon=0.5, initial_guess=nothing)
    model = Model(Ipopt.Optimizer);
    N = length(d1);
    @variable(model, fij[1:N, 1:N] >= 0.0);
    @constraint(model, non_trivial_flow, ones(1,N)*fij .== transpose(d2))
    @constraint(model, margin_flow_constraint, fij*ones(N) .<= d1);
    @objective(model, Min, sum(dist .* fij));
    function obj(sol)
        return sum(dist .* sol)
    end
    if (initial_guess == nothing)
        set_start_value.(fij, d1 ./ d2)
    else
        set_start_value.(fij, initial_guess)
        println("initial guess loss = $(obj(initial_guess))")
    end
    optimize!(model);

    println("final solution loss = $(obj(value.(fij)))")
    return model;
end
