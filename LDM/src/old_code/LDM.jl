module LDM
using Reexport;

@reexport using Plots;
@reexport using Printf;
@reexport using PyCall;
@reexport using BenchmarkTools;
@reexport using SpecialFunctions;
@reexport using ImageFiltering;
@reexport using LinearAlgebra;
@reexport using Flux;
@reexport using Statistics, StatsBase;

const RealArray{N} = Array{T,N} where {T<:Real};
const FPArray{N} = Array{T,N} where {T<:AbstractFloat};

include("../types/LagrangianDataset.jl");
include("../types/EularianDataset.jl");

include("./dataUtils.jl");
include("./TBNN.jl");
include("./FFNetwork.jl");
include("./Parashar.jl");
include("./RDGF.jl");
include("./RDTBNN.jl");
include("./postProcessing.jl");

function runTests()
    include("../test/runAllTests.jl");
end
end
