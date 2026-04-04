module RDGF

using Plots;
using Printf;
using PyCall;
using BenchmarkTools;
using SpecialFunctions;
using ImageFiltering;
using LinearAlgebra;
using Flux;
using Statistics, StatsBase;
using LaTeXStrings;
using Serialization;

const RealArray{N} = Array{T,N} where {T<:Real};
const FPArray{N} = Array{T,N} where {T<:AbstractFloat};

include("../types/LagrangianDataset.jl");
include("postProcessing.jl");

function predictPHFromVGT(vgt::FPArray{4}, timepoint, dt)
    return predictPHFromVGT(vgt[:,:,timepoint,:], dt);
end

function predictPHFromVGT(vgt::FPArray{3}, dt)
    numSamples = size(vgt)[end];
    ph = zeros(3,3,numSamples);

    alpha = -(2/7);
    beta  = -(2/5);
    gamma = 0.063;
    
    Threads.@threads for i in 1:numSamples
        Q = -0.5*(tr(vgt[:,:,i]^2));
        S = 0.5*(vgt[:,:,i] + vgt[:,:,i]');
        W = 0.5*(vgt[:,:,i] - vgt[:,:,i]');
        Dinv = exp(-dt*vgt[:,:,i]);
        Cinv = (Dinv'*Dinv);

        T2 = S*W-W*S;
        T3 = S^2 - (tr(S^2)/3)*I;
        T4 = W^2 - (tr(W^2)/3)*I;
        
        P_d = alpha*T3 + beta*T4 + gamma*T2;
        G = Dinv'*P_d*Dinv;
        ph[:,:,i] .= (2*Q/tr(Cinv) - (tr(G)/tr(Cinv)))*Cinv + G;
        ph[:,:,i] -= (1/3)*tr(ph[:,:,i])*I; #meant to predict the traceless portion
    end

    return ph;
end

# grad w.r.t. dt
function lossAndGrads_dt(vgt::FPArray{3}, gtPH::FPArray{3}, dt)
    @assert size(vgt) == size(gtPH);

    numSamples = size(vgt)[end];

    alpha = -(2/7);
    beta  = -(2/5);
    gamma = 0.063;

    loss = zeros(numSamples);
    grad = zeros(numSamples);
    
    Threads.@threads for i in 1:numSamples
        A = vgt[:,:,i]
        Q = -0.5*(tr(A^2));
        S = 0.5*(A + A');
        W = 0.5*(A - A');
        Dinv = exp(-dt*A);
        Cinv = (Dinv'*Dinv);

        T2 = S*W-W*S;
        T3 = S^2 - (tr(S^2)/3)*I;
        T4 = W^2 - (tr(W^2)/3)*I;
        
        P_d = alpha*T3 + beta*T4 + gamma*T2;
        G = Dinv'*P_d*Dinv;
        predPH = (2*Q/tr(Cinv) - (tr(G)/tr(Cinv)))*Cinv + G;
        predPH -= (1/3)*tr(predPH)*I; #meant to predict the traceless portion
        δ = gtPH[:,:,i] .- predPH;
        ξ = (Dinv')*A*Dinv + Dinv*(A')*(Dinv');

        loss[i] = norm(δ,2)^2;
        intermed = δ*(1/(tr(Cinv)^2))*(tr(Cinv)*ξ - tr(ξ)*Cinv)
        grad[i] = -2*(2*Q-tr(G))*sum(intermed);
    end

    return sum(loss)/numSamples, sum(grad)/numSamples;
end

function runRDGF(inputPath,
                 outputPath;
                 dt = 3e-4,
                 learningRate = 5e-5,
                 maxIters = 100)
    data = LagrangianDataset(deserialize(inputPath));
    N = 100000;
    trainData = LagrangianDataset(data.vgt[:,:,:,1:N],
                                  data.ph[:,:,:,1:N],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);

    testData = LagrangianDataset(data.vgt[:,:,:,N+1:2N],
                                  data.ph[:,:,:,N+1:2N],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);

    opt = Descent(learningRate)
    lossArray = zeros(2,maxIters);
    params = [dt];
    for i in 1:maxIters
        l,g = lossAndGrads_dt(trainData.vgt[:,:,1,:],
                              trainData.ph[:,:,1,:],
                              params[1]);
        lossArray[1,i] = l;
        lossArray[2,i] = lossAndGrads_dt(testData.vgt[:,:,1,:], testData.ph[:,:,1,:],params[1])[1];
        Flux.Optimise.update!(opt, params, [g]);
        display(l);
    end

    predPH = predictPHFromVGT(testData.vgt[:,:,1,:], dt);

    postProcessCommon(testData.vgt[:,:,1,:],
                      testData.ph[:,:,1,:],
                      predPH,
                      outputPath,
                      lossArray);

    display(params[1]);

end

end
