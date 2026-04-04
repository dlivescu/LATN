struct EularianDataset
    u::Array{Float64, 3};
    v::Array{Float64, 3};
    w::Array{Float64, 3};
    p::Array{Float64, 3};
    dt::Float64;
    Re::Float64;
    L::Float64;
    NX::Int;
    NY::Int;
    NZ::Int;
    pathToSource::String;
end
