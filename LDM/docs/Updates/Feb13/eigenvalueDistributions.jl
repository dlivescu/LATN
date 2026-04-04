basepath = "/home/cmhyett/Work/LDM/";
using Pkg;
Pkg.activate(basepath);
Pkg.instantiate();
using LDM;
using LinearAlgebra, Serialization, Plots;

datapath = basepath * "/results/lanl240/lanl_240_vgt.jls";
vgt = deserialize(datapath);
_,_,numSamples = size(vgt);

S_eigValArr = zeros(3,numSamples);

Threads.@threads for i in 1:numSamples
    A = vgt[:,:,i];
    S = 0.5*(A+A');
    W = 0.5*(A-A');
    
    S_eigValArr[:,i] .= eigen(S).values;
end

invars = LDM.TBNN.calcInvariants(vgt);

