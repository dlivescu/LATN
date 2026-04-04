#= 
in this file I'd like to spin up synthetic data to test the gradient calculations. 

I.e., given a set of velocity gradient tensors vgt of size (3,3,N), generate mock pressure hessian data that is a linear combination of Tensor basis and a known function of invariants.

In particular, given a 10x5 matrix W, let
mockPH = W_ij*x_j*TB_i
=#

using Flux, Test;

include("../src/LagrangianDeformationModels.jl");

function generateMockPH(invariants, tb; nn, p)
    numSamples = size(invariants)[end];
    mockPH = zeros(3,3,numSamples);
    coefs = nn(p)(invariants);
    
    for i in 1:numSamples
        mockPH[:,:,i] = sum([coefs[j,i]*tb[j,i] for j in 1:10]);
    end

    return mockPH;
end

numSamples = 5;
vgt = rand(3, 3, numSamples);
Threads.@threads for i in 1:numSamples
    vgt[3,3,i] = -(vgt[1,1,i] + vgt[2,2,i]);
end
tb = fill(zeros(3,3), (10,numSamples));
invariants = zeros(5,numSamples);
LDM.TBNN.calcSymmetricTensorBasis!(tb, vgt);
LDM.TBNN.calcInvariants!(invariants, vgt);

@testset "trivial gradient check" begin
    W = zeros(Float32, 10,5);
    ph = zeros(Float32, 3,3,numSamples);
    p, nn = Flux.destructure(Chain(Dense(LDM.TBNN.numInvariants, LDM.TBNN.numBasisElements)));
    p[1:50] .= reshape(W, prod(size(W)));
    p[51:end] .= 0.0;
    l, grads = LDM.TBNN.lossAndGrads(invariants, ph, tb, nn, p, [i for i in 1:numSamples]);

    for i in 1:length(grads)
        @test grads[i] .≈ 0.0;
    end
end

@testset "linear function gradient check" begin
    W = rand(10,5);
    p, nn = Flux.destructure(Chain(Dense(LDM.TBNN.numInvariants, LDM.TBNN.numBasisElements)));
    p[1:50] .= reshape(W, prod(size(W)));
    p[51:end] .= 0.0;
    ph = generateMockPH(invariants, tb, nn=nn, p=p);
    l, grads = LDM.TBNN.lossAndGrads(invariants, ph, tb, nn, p, [i for i in 1:numSamples]);

    for i in 1:length(grads)
        @test isapprox(grads[i], 0.0, atol=10*eps(Float32));
    end
end

@testset "nonlinear function gradient check" begin
    hiddenDim = 50;
    p, nn = Flux.destructure(Chain(Dense(LDM.TBNN.numInvariants, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, LDM.TBNN.numBasisElements)));
    ph = generateMockPH(invariants, tb, nn=nn, p=p);
    l, grads = LDM.TBNN.lossAndGrads(invariants, ph, tb, nn, p, [i for i in 1:numSamples]);

    for i in 1:length(grads)
        @test isapprox(grads[i], 0.0, atol=10*eps(Float32));
    end
end
