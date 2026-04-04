module RDTBNN

using Plots;
using Printf;
using PyCall;
using BenchmarkTools;
using SpecialFunctions;
using ImageFiltering;
using LinearAlgebra;
using Flux, Zygote;
using Statistics, StatsBase;
using LaTeXStrings;
using Serialization;

const RealArray{N} = Array{T,N} where {T<:Real};
const FPArray{N} = Array{T,N} where {T<:AbstractFloat};

include("tbnnUtils.jl");
include("../types/LagrangianDataset.jl");
include("postProcessing.jl");

function predictPressureHessian(normalizedVGT::FPArray{2}, nn, p)
    tb = calcSymmetricTensorBasis(normalizedVGT);
    invariants = calcInvariants(normalizedVGT);

    predCoefs = nn(p)(invariants);
    intermed = [predCoefs[i]*tb[i] for i in 1:numBasisElements];
    H = sum(intermed);
    Dinv = (I + normalizedVGT);
    return Dinv'*H*Dinv;
end

function predictPressureHessian(normalizedVGT::FPArray{3}, nn, p)
    numSamples = size(normalizedVGT)[end];
    predPH = zeros(3,3,numSamples);

    tb = calcSymmetricTensorBasis(normalizedVGT);
    invariants = calcInvariants(normalizedVGT);
    predCoefs = nn(p)(invariants);
    intermediate = [[predCoefs[i,j]*tb[i,j] for i in 1:numBasisElements] for j in 1:numSamples];

    Threads.@threads for j in 1:numSamples
        H = sum(intermediate[j]); #sum across TB to generate upstream PH
        A = normalizedVGT[:,:,i];
        Dinv = (I + A); #linear approximation of exp(A)
        predPH[:,:,j] .= Dinv'*H*Dinv;
    end

    return predPH;
end

function predictPressureHessian(invars, tb, nn, p, inds)
    numSamples = size(inds)[end];
    predCoefs = nn(p)(invars[:,inds]);
    intermediate = [[predCoefs[i,j]*tb[i,inds[j]] for i in 1:numBasisElements] for j in 1:numSamples];
    predictedPressureHessian = zeros(3,3,numSamples);
    Threads.@threads for i in 1:numSamples
        predictedPressureHessian[:,:,i] .= sum(intermediate[i]);
    end
    return predictedPressureHessian;
end


function loss(normalizedVGT::FPArray{3}, normalizedPH::FPArray{3}, nn, p)
    numSamples = size(normalizedVGT)[end];
    predPH = [predictPressureHessian(normalizedVGT[:,:,i], nn, p) for i in 1:numSamples];
    δ = [(normalizedPH[:,:,i] - predPH[i]) for i in 1:numSamples];
    loss = (1/numSamples)*sum([norm(δ[i])^2 for i in 1:numSamples]);
    return loss;
end

function lossAndGrads(normalizedVGT::FPArray{3},
                      normalizedPH::FPArray{3},
                      tb,
                      invars,
                      nn, p,
                      inds)
    N = length(inds);
    M = length(p);
    dHdp = zeros(3,3,N,M);
    dΔ = zeros(3,3,N,M);
    dL = zeros(M);
    Δ = zeros(3,3,N);

    H_tb = predictPressureHessian(invars, tb, nn, p, inds);
    
    df(θ, i) = jacobian(ps->nn(ps)(invars[:,i]), θ)[1];

    Threads.@threads for n in 1:N
        A = normalizedVGT[:,:,n]
        d = df(p,n);
        Δ[:,:,n] .= normalizedPH[:,:,n] - (I + A)'*H_tb[:,:,n]*(I+A);
        for m in 1:M
            for j in 1:numBasisElements
                dHdp[:,:,n,m] .+= tb[j,n]*d[j,m];
            end
            dΔ[:,:,n,m] = -dHdp[:,:,n,m] - A'*dHdp[:,:,n,m] - dHdp[:,:,n,m]*A - A'*dHdp[:,:,n,m]*A;
        end
    end

    loss = (1/N)*sum([norm(Δ[:,:,n])^2 for n in 1:N]);
    Threads.@threads for m in 1:M
        for n in 1:N
            for i in 1:3
                for j in 1:3
                    dL[m] += Δ[i,j,n]*dΔ[i,j,n,m];
                end
            end
        end
    end
    dL .*= -2/N;

    return loss, dL;
end

function train!(trainData::LagrangianDataset, testData::LagrangianDataset, nn, params; timestep=1, maxiters=500, learningRate=5e-3)
    numSamples = size(trainData.vgt)[end];

    timescale = calcCharacteristicTimescale(trainData.vgt[:,:,timestep,i]);
    normalizedVGT = trainData.vgt[:,:,timestep,:] * timescale;
    normalizedPH = testData.ph[:,:,timestep,:] * timescale^2;

    #setup test data
    numTestSamples = size(testData.vgt)[end];
    normalizedVGT_test = testData.vgt[:,:,timestep,:] * timescale;
    normalizedPH_test  = testData.ph[:,:,timestep,:] * timescale^2;

    opt = ADAM(learningRate);
    lossArray = zeros(2,maxiters);
    for i in 1:maxiters
        dl(θ) = gradient(params->loss(normalizedVGT,
                                      normalizedPH,
                                      nn,
                                      params), θ)[1];
        lossArray[1,i] = loss(normalizedVGT, normalizedPH, nn, params);
        lossArray[2,i] = loss(normalizedVGT_test, normalizedPH_test, nn, params);
        g = dl(params);
        Flux.Optimise.update!(opt, params, g);
        display(lossArray[1,i]);
    end

    return lossArray, params;
end

function newTrain!(trainData::LagrangianDataset, testData::LagrangianDataset, nn, params, inds; timestep=1, maxiters=500, learningRate=5e-3, miniBatchSize=100)
    numSamples = size(trainData.vgt)[end];

    timescale = calcCharacteristicTimescale(trainData.vgt[:,:,timestep,i]);
    normalizedVGT = trainData.vgt[:,:,timestep,:] * timescale;
    normalizedPH = testData.ph[:,:,timestep,:] * timescale^2;

    tb = calcSymmetricTensorBasis(normalizedVGT);
    invars = calcInvariants(normalizedVGT);

    #setup test data
    numTestSamples = size(testData.vgt)[end];
    normalizedVGT_test = testData.vgt[:,:,timestep,:] * timescale;
    normalizedPH_test  = testData.ph[:,:,timestep,:] * timescale^2;

    opt = ADAM(learningRate);
    lossArray = zeros(2,maxiters);
    N = length(inds);
    for i in 1:maxiters
        sampledInds = sample(1:N, N, replace=false);
        numItersPerBatch = floor(Int, N/miniBatchSize);
        miniBatchInds = [sampledInds[i:i+miniBatchSize-1] for i in 1:miniBatchSize:N-miniBatchSize+1];
        l = 0;
        for j in 1:numItersPerBatch
            l, g = lossAndGrads(normalizedVGT,
                                normalizedPH,
                                tb,
                                invars,
                                nn,
                                params,
                                miniBatchInds[j]);
            Flux.Optimise.update!(opt, params, g);
            lossArray[1,i] += l;
            print("iteration loss = $(l)\n");
        end
        print("epoch loss = $(lossArray[1,i])\n");
    end

    return lossArray, params;
end

function predictPHFromVGT(vgt::FPArray{4}, nn, params, timepoint)
    return reshape(predictPHFromVGT(vgt[:,:,timepoint,:], nn, params), (3,3,1,size(vgt)[end]));
end

function predictPHFromVGT(vgt::FPArray{3}, nn, params)
    numSamples = size(vgt)[end];
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    tb = calcSymmetricTensorBasis(normalizedVGT);
    invars = calcInvariants(normalizedVGT);

    ph = predictPressureHessian(invars, tb, nn, params, [i for i in 1:numSamples])/timescale^2;
    for i in 1:numSamples
        ph[:,:,i] -= (tr(ph[:,:,i])/3)*Matrix(I,3,3);
    end
    return ph;
end

function run(inputPath, outputPath; numTrainTrajs = 10000, hiddenDim=30, maxiters=500, learningRate=5e-3, miniBatchSize=500)
    data = deserialize(inputPath);

    N = numTrainTrajs;
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

    params, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                        Dense(hiddenDim, hiddenDim, tanh),
                                        Dense(hiddenDim, hiddenDim, tanh),
                                        Dense(hiddenDim, hiddenDim, tanh),
                                        Dense(hiddenDim, hiddenDim, tanh),
                                        Dense(hiddenDim, numBasisElements)));

    lossArray, _ = newTrain!(trainData, testData, nn, params, [i for i in 1:N], maxiters=maxiters, learningRate=learningRate, miniBatchSize=miniBatchSize)

    postProcessTBNN(testData.vgt[:,:,1,:],
                    testData.ph[:,:,1,:],
                    outputPath,
                    lossArray,
                    params,
                    nn);

end

end
