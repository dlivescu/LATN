
struct LagrangianDataset
    vgt::Array{Float32, 4};
    ph::Array{Float32, 4};
    vis::Array{Float32, 4};
    sgs::Array{Float32, 4};
    pathToSource::String;
    regenerate::Function;
    filtersize::Float64;
    dt::Float64;
end

function LagrangianDataset(oldLDM)
    return LagrangianDataset(oldLDM.vgt,
                             oldLDM.ph,
                             oldLDM.vis,
                             oldLDM.sgs,
                             oldLDM.pathToSource,
                             oldLDM.regenerate,
                             oldLDM.filtersize,
                             oldLDM.dt);
end

