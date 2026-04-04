module FF

using Plots;
using Printf;
using PyCall;
using BenchmarkTools;
using SpecialFunctions;
using ImageFiltering;
using LinearAlgebra;
using Flux;
using Statistics, StatsBase;
using Serialization;

const RealArray{N} = Array{T,N} where {T<:Real};
const FPArray{N} = Array{T,N} where {T<:AbstractFloat};

include("../types/LagrangianDataset.jl");
include("postProcessing.jl");

function trainFF!(data::LagrangianDataset, nn, params;
                  maxiters = 500,
                  learningRate = 5e-3,
                  timestep = 1)

    numSamples = size(data.vgt)[end];
    timescale = calcCharacteristicTimescale(data.vgt[:,:,timestep,:]);
    normalizedVGT = data.vgt[:,:,timestep,:] * timescale;
    normalizedPH = data.ph[:,:,timestep,:] * timescale^2;

    flattenedVGT = reshape(normalizedVGT, (9,numSamples));
    deconstructedPH = zeros(6,numSamples);
    Threads.@threads for i in 1:numSamples
        deconstructedPH[1,i] = normalizedPH[1,1,i];
        deconstructedPH[2,i] = normalizedPH[1,2,i];
        deconstructedPH[3,i] = normalizedPH[1,3,i];
        deconstructedPH[4,i] = normalizedPH[2,2,i];
        deconstructedPH[5,i] = normalizedPH[2,3,i];
        deconstructedPH[6,i] = normalizedPH[3,3,i];
    end

    function loss(batchX, batchY, p)
        _numSamples = size(batchX)[end];
        predPH = nn(p)(batchX);
        return Flux.Losses.mse(predPH, batchY);
    end

    opt = ADAM(learningRate);
    lossArray = zeros(maxiters);
    batchX = flattenedVGT;
    batchY = deconstructedPH;
    trainVar = var([norm(deconstructedPH[:,i]) for i in 1:numSamples]);
    print("train variance = $(trainVar)\n");
    print("no model loss = $(Flux.Losses.mse(zeros(size(batchY)), batchY))\n");
    
    for i in 1:maxiters
        l = loss(batchX, batchY, params);
        g = Flux.gradient(p->loss(batchX, batchY, p), params);
        Flux.Optimise.update!(opt, params, g[1]);
        lossArray[i] = l;
        display(l);
    end

    return lossArray, params;
end

function predictPressureHessian(batchX, #vgt of size (9,numSamples)
                                nn, p)
    _numSamples = size(batchX)[end];
    predPH = zeros(3,3,_numSamples);
    predCoefs = nn(p)(batchX);
    Threads.@threads for i in 1:_numSamples
        predPH[1,1,i] = predCoefs[1,i];
        predPH[1,2,i] = predCoefs[2,i];
        predPH[1,3,i] = predCoefs[3,i];
        predPH[2,2,i] = predCoefs[4,i];
        predPH[2,3,i] = predCoefs[5,i];
        predPH[3,3,i] = predCoefs[6,i];
        
        predPH[2,1,i] = predPH[1,2,i];
        predPH[3,1,i] = predPH[1,3,i];
        predPH[3,2,i] = predPH[2,3,i];
        # predPH[3,3,i] = -(predPH[1,1,i] + predPH[2,2,i]);
    end

    return predPH;
end

function predictPHFromVGT(vgt, #(3,3,_numSamples)
                          nn, p)
    _numSamples = size(vgt)[end];
    return predictPressureHessian(reshape(vgt, (9,_numSamples)),
                                  nn, p);
end

function runFF(inputPath,
               outputPath;
               hiddenDim = 30,
               maxiters = 500,
               learningRate = 5e-3)

    p, nn = Flux.destructure(Chain(Dense(9,hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, 6)));

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

    testData =  LagrangianDataset(data.vgt[:,:,:,N+1:2*N],
                                  data.ph[:,:,:,N+1:2*N],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);

    lossArray, _ = trainFF!(trainData, nn, p, maxiters=maxiters, learningRate=learningRate);

    postProcessCommon(testData.vgt[:,:,1,:],
                      testData.ph[:,:,1,:],
                      outputPath,
                      lossArray,
                      p,
                      nn);
end


end
