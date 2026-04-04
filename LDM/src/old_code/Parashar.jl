module Parashar

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

const c1 = -0.0023;
const c2 = 0.2460;
const c3 = -0.1049;
const c4 = -0.0400;
const c5 = -0.0007;
const c6 = 0.5098;
const c7 = -0.6009;
const c8 = 0.8583;
const c9 = 0.3299;
const c10 = -0.0764;
const coefArray = [c1 c2 c3 c4 c5 c6 c7 c8 c9 c10];

include("tbnnUtils.jl");
include("runTBNN.jl");
include("../types/LagrangianDataset.jl");
include("postProcessing.jl");

const Gp = [-26.9693 -13.2321 -12.5971
            -13.2321 -26.0595 -17.4419
            -12.5971 -17.4419 -24.2304];

const Fp = [19.1816 23.7427 17.5106
            23.7427 27.9531 25.3751
            17.5106 25.3751 20.1042];

function loss(batchX, batchY, tb, sampleIndxs)
    _numSamples = size(batchX)[end];
    predictedPressureHessian = predictNormalizedPressureHessian(tb, sampleIndxs);
    δ = [(batchY[:,:,i] - predictedPressureHessian[:,:,i]) for i in 1:_numSamples];
    loss = sum([norm(δ[i])^2 for i in 1:_numSamples])/(_numSamples);
    return loss;
end

function calcMinMaxElementsMatrices(tb, ph)
    @assert size(tb)[end] == size(ph)[end];
    numSamples = size(tb)[end];

    function calcMinMax(mat, ind::CartesianIndex; f=minimum)
        numSamples = length(mat);
        arr = [mat[i][ind] for i in 1:numSamples];
        return f(arr);
    end

    Gs = [zeros(3,3) for i in 1:(numBasisElements+1)];
    Fs = [zeros(3,3) for i in 1:(numBasisElements+1)];

    for k in 1:numBasisElements
        for i in 1:3
            for j in 1:3
                Gs[k][i,j] = calcMinMax(tb[k,:], CartesianIndex(i,j), f=minimum);
                Fs[k][i,j] = calcMinMax(tb[k,:], CartesianIndex(i,j), f=maximum);
            end
        end
    end

    for i in 1:3
        for j in 1:3
            Gs[end][i,j] = minimum(ph[i,j,:]);
            Fs[end][i,j] = maximum(ph[i,j,:]);
        end
    end

    return Fs, Gs;
end


function trainParasharTB(data::LagrangianDataset; timestep=1)
    numSamples = size(data.vgt)[end];
    invars = zeros(numInvariants, numSamples);
    tb = [zeros(3,3) for i in 1:numBasisElements*numSamples];
    tb = reshape(tb, (numBasisElements, numSamples));
    
    epsilon = mean([norm(data.vgt[:,:,timestep,i]) for i in 1:numSamples]);
    normalizedVGT = data.vgt[:,:,timestep,:] ./ epsilon;
    normalizedPH = data.ph[:,:,timestep,:] ./ (epsilon^2);

    calcSymmetricTensorBasis!(tb, normalizedVGT);

    Fs, Gs = calcMinMaxElementsMatrices(tb, normalizedPH);

    return Fs, Gs;
end


function predictPHFromVGT(vgt::FPArray{4}, Fs, Gs, timepoint)
    numSamples = size(vgt)[end];
    ph = predictPHFromVGT(data.vgt[:,:,timestep,:]);
    return reshape(ph, (3,3,1,numSamples));
end

function predictPHFromVGT(vgt::FPArray{3}, Fs, Gs)
    numSamples = size(vgt)[end];

    # non-dimensionalize VGT using avg Frobenius norm
    epsilon = mean([norm(vgt[:,:,i]) for i in 1:numSamples]);
    normalizedVGT = vgt / epsilon;
    
    # calculate tensor basis using normalized VGT
    tb = [zeros(3,3) for i in 1:(numBasisElements*numSamples)];
    tb = reshape(tb, (numBasisElements, numSamples));
    calcSymmetricTensorBasis!(tb, normalizedVGT);

    # min-max TB
    for i in 1:numBasisElements
        for j in 1:numSamples
            tb[i,j] .= (tb[i,j] - Gs[i]) ./ (Fs[i]-Gs[i]);
        end
    end

    # take linear combination of TB using mean coefficients
    intermediate = [[coefArray[i]*tb[i,j] for i in 1:numBasisElements] for j in 1:numSamples];
    ph = zeros(3,3,numSamples);
    Threads.@threads for i in 1:numSamples
        ph[:,:,i] .= sum(intermediate[i]);
    end

    #re-dimensionalize and enforce traceless
    for i in 1:numSamples
        ph[:,:,i] = (ph[:,:,i] .* (Fs[end] - Gs[end]) + Gs[end]); #un-min-max
        ph[:,:,i] .*= epsilon^2; #un-normalize
        ph[:,:,i] = ph[:,:,i] - (tr(ph[:,:,i])/3)*I; # remove trace
    end

    return ph;
end

function runParashar(inputPath,
                     outputPath)
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
    testData = trainData;
    # testData =  LagrangianDataset(data.vgt[:,:,:,N+1:2*N],
    #                               data.ph[:,:,:,N+1:2*N],
    #                               data.vis,
    #                               data.sgs,
    #                               data.pathToSource,
    #                               data.regenerate,
    #                               data.filtersize,
    #                               data.dt);

    Fs, Gs = setFsGs();#trainParasharTB(trainData, timestep=1)

    postProcessParashar(testData.vgt[:,:,1,:],
                        testData.ph[:,:,1,:],
                        outputPath,
                        Fs,
                        Gs);

end

function setFsGs()
    G1 = [-3.5709 -3.0858 -2.7680
          -3.0858 -3.0346 -3.0692
          -2.7680 -3.0692 -4.2937]
    G2 = [-25.6104 -31.8796 -21.5908
          -31.8796 -30.5056 -22.2121
          -21.5908 -22.2121 -28.0164]
    G3 = [-9.2866 -6.4438 -10.5787
          -6.4438 -6.4369 -5.4972
          -10.5787 -5.4972 -8.4430]
    G4 = [-14.8512 -15.3252 -13.8140
          -15.3252 -11.2078 -11.7173
          -13.8140 -11.7173 -13.0263]
    G5 = [-54.1672 -26.9697 -23.7045
          -26.9697 -43.5902 -36.9865
          -23.7045 -36.9865 -30.1412]
    G6 = [-119.1511 -78.0652 -112.3875
          -78.0652 -186.4133 -80.4339
          -112.3875 -80.4339 -116.5743]
    G7 = [-737.6056 -1038.4397 -640.3922
          -1038.4397 -963.7453 -551.0073
          -640.3922 -551.0073 -537.7603]
    G8 = [-496.1792 -376.4132 -245.1099
          -376.4132 -341.9391 -345.2818
          -245.1099 -345.2818 -405.4714]
    G9 = [-526.6148 -345.7790 -276.1035
          -345.7790 -188.0288 -293.5551
          -276.1035 -293.5551 -339.0242]
    G10 = [-595.3592 -1679.5692 -922.0738
           -1679.5692 -1466.6689 -674.1032
           -922.0738 -674.1032 -1079.9475]
    Gp = [-26.9693 -13.2321 -12.5971
          -13.2321 -26.0595 -17.4419
          -12.5971 -17.4419 -24.2304]
    F1 = [2.7021 3.2914 2.3884
          3.2914 3.3876 2.8077
          2.3884 2.8077 2.8905]
    F2 = [32.0154 35.4956 20.6083
          35.4956 29.8806 30.5352
          20.6083 30.5352 31.7193]
    F3 = [8.2153 7.9086 7.8492
          7.9086 6.0971 5.9908
          7.8492 5.9908 9.3168]
    F4 = [16.6632 9.6823 12.9011
          9.6823 27.8775 15.5748
          12.9011 15.5748 23.0473]
    F5 = [27.5052 57.6332 33.7410
          57.6332 53.5055 38.4958
          33.7410 38.4958 55.2717]
    F6 = [195.0304 143.0951 108.0702
          143.0951 92.4415 169.8799
          108.0702 169.8799 135.2173]
    F7 = [780.8498 970.6533 540.2198
          970.6533 808.6133 523.5790
          540.2198 523.5790 924.0598]
    F8 = [390.7658 778.0012 298.8682
          778.0012 658.3773 509.5283
          298.8682 509.5283 444.1750]
    F9 = [381.9239 194.5271 578.1185
          194.5271 197.7692 170.3597
          578.1185 170.3597 598.0897]
    F10 = [1335.8832 1173.0048 438.1600
           1173.0048 1126.0351 671.8583
           438.1600 671.8582 662.4745]
    Fp = [19.1816 23.7427 17.5106
          23.7427 27.9531 25.3751
          17.5106 25.3751 20.1042]

    function issymmetric(mat)
        return (mat[1,2] == mat[2,1] &&
                mat[1,3] == mat[3,1] &&
                mat[2,3] == mat[2,3])
    end

    Fs = [F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, Fp]
    Gs = [G1, G2, G3, G4, G5, G6, G7, G8, G9, G10, Gp]

    for i in 1:10
        @assert issymmetric(Fs[i])
        @assert issymmetric(Gs[i])
    end
    
    return Fs, Gs;
end

end
