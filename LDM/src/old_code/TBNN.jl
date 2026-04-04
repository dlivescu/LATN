module TBNN

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

const RealArray{N} = Array{T,N} where {T<:Real};
const FPArray{N} = Array{T,N} where {T<:AbstractFloat};

include("tbnnUtils.jl");
include("runTBNN.jl");
include("../types/LagrangianDataset.jl");
include("postProcessing.jl");

function predictPressureHessian(batchX, tb, nn, p, sampleIndxs)
    _numSamples = size(batchX)[end];
    predCoefs = nn(p)(batchX);
    intermediate = [[predCoefs[i,j]*tb[i,sampleIndxs[j]] for i in 1:numBasisElements] for j in 1:_numSamples];
    predictedPressureHessian = zeros(3,3,_numSamples);
    Threads.@threads for i in 1:_numSamples
        predictedPressureHessian[:,:,i] .= sum(intermediate[i]);
    end
    return predictedPressureHessian;
end
function lossAndGrads(batchX, batchY, tb, nn, p, sampleIndxs, trainVar)
    _numSamples = size(batchX)[end];
    predictedPressureHessian = predictPressureHessian(batchX, tb, nn, p, sampleIndxs);
    δ = [(batchY[:,:,i] - predictedPressureHessian[:,:,i]) for i in 1:_numSamples];
    contraction = hcat([[sum(δ[i].*tb[j,i]) for i in 1:_numSamples] for j in 1:numBasisElements]...);
    grads = -(2/(_numSamples))*Flux.pullback(ps->nn(ps)(batchX),p)[2](contraction')[1];
    loss = sum([norm(δ[i])^2 for i in 1:_numSamples])/(_numSamples);
    return loss/trainVar,grads;
end
function loss(batchX, batchY, tb, nn, p, sampleIndxs, trainVar)
    _numSamples = size(batchX)[end];
    predictedPressureHessian = predictPressureHessian(batchX, tb, nn, p, sampleIndxs);
    δ = [(batchY[:,:,i] - predictedPressureHessian[:,:,i]) for i in 1:_numSamples];
    loss = sum([norm(δ[i])^2 for i in 1:_numSamples])/(_numSamples);
    return loss/trainVar;
end


# returns prediction of pressure hessians based on inputData and parameterized NN
function trainTB!(data::LagrangianDataset, testData::LagrangianDataset, nn, params; timestep=1, maxiters=500, learningRate = 2e1)
    numSamples = size(data.vgt)[end];
    invars = zeros(numInvariants, numSamples);
    tb = [zeros(3,3) for i in 1:numBasisElements*numSamples];
    tb = reshape(tb, (numBasisElements, numSamples));

    timescale = calcCharacteristicTimescale(data.vgt[:,:,timestep,:]);
    normalizedVGT = data.vgt[:,:,timestep,:] * timescale;
    normalizedPH = data.ph[:,:,timestep,:] * timescale^2;
    calcInvariants!(invars, normalizedVGT);
    calcSymmetricTensorBasis!(tb, normalizedVGT);

    #setup test data
    numTestSamples = size(testData.vgt)[end];
    normalizedVGT_test = testData.vgt[:,:,timestep,:] * timescale;
    normalizedPH_test = testData.ph[:,:,timestep,:] * timescale^2;
    invars_test = zeros(numInvariants, numTestSamples);
    tb_test = fill(zeros(3,3), (numBasisElements, numTestSamples));
    calcInvariants!(invars_test, normalizedVGT_test);
    calcSymmetricTensorBasis!(tb_test, normalizedVGT_test);
    
    trainVar = var([norm(normalizedPH[:,:,i]) for i in 1:numSamples]);
    print("train variance = $(trainVar)\n");

    opt = ADAM(learningRate, (0.9, 0.999), 1e-6);
    lossArray = zeros(2,maxiters);
    etaArray = zeros(maxiters);
    inds = [i for i in 1:numSamples];
    testInds = [i for i in 1:numTestSamples];
    returnParams = zeros(size(params));
    minTestLoss = Inf;
    for i in 1:maxiters
        l,g = lossAndGrads(invars, normalizedPH, tb, nn, params, inds, trainVar);
        lossArray[1,i] = l;
        lossArray[2,i] = loss(invars_test, normalizedPH_test, tb_test, nn, params, testInds, trainVar);
        Flux.Optimise.update!(opt, params, g);
        if (lossArray[2,i] < minTestLoss)
            returnParams .= copy(params);
            minTestLoss = lossArray[2,i];
        end
        println("$(l)");
    end

    return lossArray, returnParams;
end

function predictPHFromVGT(vgt::FPArray{4}, nn, params, timepoint)
    numSamples = size(vgt)[end];

    timescale = calcCharacteristicTimescale(vgt[:,:,timepoint,:]);
    normalizedVGT = vgt * timescale;
    
    invars = zeros(numInvariants, numSamples);
    tb = fill(zeros(3,3), (numBasisElements, numSamples));
    calcInvariants!(invars, normalizedVGT[:,:,timepoint,:]);
    calcSymmetricTensorBasis!(tb, normalizedVGT[:,:,timepoint,:]);

    ph = predictPressureHessian(invars, tb, nn, params, [i for i in 1:numSamples]) / timescale^2;
    return reshape(ph, (3,3,1,numSamples));
end

function predictPHFromVGT(vgt::FPArray{3}, nn, params)
    numSamples = size(vgt)[end];
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    
    invars = zeros(numInvariants, numSamples);
    tb = fill(zeros(3,3), (numBasisElements, numSamples));
    calcInvariants!(invars, normalizedVGT);
    calcSymmetricTensorBasis!(tb, normalizedVGT);

    ph = predictPressureHessian(invars, tb, nn, params, [i for i in 1:numSamples]) / timescale^2;
    return ph;
end

function predictPHFromVGT_latent(vgt::FPArray{3}, nn, params)
    numSamples = size(vgt)[end];
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    
    invars = zeros(numInvariants, numSamples);
    tb = fill(zeros(3,3), (numBasisElements, numSamples));
    calcInvariants!(invars, normalizedVGT);
    calcSymmetricTensorBasis!(tb, normalizedVGT);
    invars[3:5,:] .= 0.0;
    
    ph = predictPressureHessian(invars, tb, nn, params, [i for i in 1:numSamples]) / timescale^2;
    return ph;
end


end
