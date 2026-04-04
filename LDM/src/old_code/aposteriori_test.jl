using PyCall, StatsBase;

basepath = "/home/cmhyett/Work/LDM/data/lanl_1024/LagrangianData/"
N = 131072;
NT = 1000;
dt = 3e-4;

py"""
import numpy as np;
vgt = np.fromfile($(basepath) + "./aij_1024_dns.bin");
pij = np.fromfile($(basepath) + "./pij_1024_dns.bin");
vis = np.fromfile($(basepath) + "./vis_1024_dns.bin");
vgt = vgt.reshape([$(N), $(NT), 3,3]);
pij = pij.reshape([$(N), $(NT), 3,3]);
vis = vis.reshape([$(N), $(NT), 3,3]);
vgt = vgt[:, 0:100, :,:];
pij = pij[:, 0:100, :,:];
vis = vis[:, 0:100, :,:];
"""

vgt = PyArray(py"vgt"o);
pij = PyArray(py"pij"o);
vis = PyArray(py"vis"o);

function pressure(vgt, t, dt, ind)
    j = floor(Int, t/dt);
    return pij[ind,j,:,:];
end

function viscosity(vgt, t, dt, ind)
    j = floor(Int, t/dt);
    return vis[ind,j,:,:];
end

@inline function interpolate(x1, x2, y1, y2, x)
    return ((y2-y1)/(x2-x1))*(x-x1) + y1;
end

function forwardEuler(vgt, dt, t, ind)
    return vgt[ind,:,:] + dt*(-vgt[ind,:,:]^2 - pressure(vgt, t, dt, ind) + viscosity(vgt, t, dt, ind));
end

function rk4(vgt, dt, t, sampleInd)
    function rhs(_t, _vgt)
        _ph = interpolate(t, t+dt, pressure(vgt, t, dt, sampleInd), pressure(_vgt, t+dt, dt, sampleInd), _t);
        _vis =interpolate(t, t+dt, viscosity(vgt, t, dt, sampleInd), viscosity(_vgt, t+dt, dt, sampleInd), _t);
        return -_vgt^2 - _ph + _vis
    end
    k1 = rhs(t,vgt[sampleInd, :,:])
    k2 = rhs(t+dt/2, vgt[sampleInd, :,:]+(dt/2)*k1);
    k3 = rhs(t+dt/2, vgt[sampleInd, :,:]+(dt/2)*k2);
    k4 = rhs(t+dt, vgt[sampleInd, :,:]+dt*k3);
    return vgt[sampleInd,:,:] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end

function advance(initialCondition, numSteps, dt; alg=forwardEuler)
    numSamples, _, _ = size(initialCondition);
    vgt = copy(initialCondition);
    for i in 1:numSteps
        Threads.@threads for j in 1:numSamples
            vgt[j,:,:] .= alg(vgt, dt, i*dt, j);
        end
    end
    return vgt;
end

function mse(x,y)
    badInds = findall(n->n==Inf,y);
    println(
    badInds = unique([badInds[i][1] for i in 1:length(badInds)]);
    goodInds = [i for i in 1:N];
    deleteat!(goodInds, badInds);
    println("num unstable trajs = $(N-length(goodInds))");
    return mean((x[goodInds,:,:] .- y[goodInds,:,:]) .^ 2)
end

function groundTruthDeviation(vgt, dt; tstep=20, maxT=300)
    numSteps = tstep:tstep:maxT;
    @time mse_loss_fe = [mse(vgt[:,i+1,:,:], advance(vgt[:,1,:,:], i, 3e-4, alg=forwardEuler)) for i in numSteps];
    #@time mse_loss_rk4 = [mse(vgt[:,i+1,:,:], advance(vgt[:,1,:,:], i, 3e-4, alg=rk4)) for i in numSteps];
    return numSteps, mse_loss_fe;
end

#x, y_fe = groundTruthDeviation(vgt, dt, tstep=99, maxT=99)
#using Plots
#plot(x, y_fe, label="forward Euler")
#plot!(x, y_rk4, label="rk4")
