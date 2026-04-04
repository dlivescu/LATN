using Serialization;

function runTBNN(inputPath,
                 outputPath;
                 hiddenDim = 30,
                 maxiters = 500,
                 learningRate = 5e-3,
                 nn = nothing,
                 p = nothing)

    if (nn == nothing && p == nothing)
        p, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, numBasisElements)));
    end

    data = LagrangianDataset(deserialize(inputPath));
    N = min(262144, floor(Int, size(data.vgt)[end]/2));
    trainData = LagrangianDataset(data.vgt[:,:,:,1:N],
                                  data.ph[:,:,:,1:N],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);
    testData =  LagrangianDataset(data.vgt[:,:,:,N+1:end],
                                  data.ph[:,:,:,N+1:end],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);


    lossArray, _ = trainTB!(trainData, testData, nn, p, maxiters=maxiters, learningRate=learningRate);

    postProcessTBNN(testData.vgt[:,:,1,:],
                    testData.ph[:,:,1,:],
                    outputPath,
                    lossArray,
                    p,
                    nn);
end

function postProcessTBNNAfterFailedRun(datapath,
                                       outputpath,
                                       parampath,
                                       losspath;
                                       hiddenDim = 30,
                                       nn = nothing)

    if (nn == nothing)
        _, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, hiddenDim, tanh),
                                       Dense(hiddenDim, numBasisElements)));
    end
    p = deserialize(parampath);

    data = LagrangianDataset(deserialize(datapath));
    N = min(262144, floor(Int, size(data.vgt)[end]/2));
    trainData = LagrangianDataset(data.vgt[:,:,:,1:N],
                                  data.ph[:,:,:,1:N],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);
    testData =  LagrangianDataset(data.vgt[:,:,:,N+1:end],
                                  data.ph[:,:,:,N+1:end],
                                  data.vis,
                                  data.sgs,
                                  data.pathToSource,
                                  data.regenerate,
                                  data.filtersize,
                                  data.dt);

    lossArray = deserialize(losspath);

    postProcessTBNN(testData.vgt[:,:,1,:],
                    testData.ph[:,:,1,:],
                    outputpath,
                    lossArray,
                    p,
                    nn);
end

function calcHistogramsOfInvariants(inputPath)
    data = LagrangianDataset(deserialize(inputPath));
    timescale = calcCharacteristicTimescale(data.vgt[:,:,1,:]);
    vgt = data.vgt[:,:,1,:];
    normalizedVGT = data.vgt[:,:,1,:]*timescale;

    normalizedInvariants = calcInvariants(normalizedVGT);
    unnormalizedInvariants = calcInvariants(vgt);

    h_normalized = [];
    h_unnormalized = [];
    for i in 1:numInvariants
        h_n_temp = fit(Histogram, normalizedInvariants[i,:], nbins=100);
        h_u_temp = fit(Histogram, unnormalizedInvariants[i,:], nbins=100);
        push!(h_normalized, normalize(h_n_temp, mode=:pdf));
        push!(h_unnormalized, normalize(h_u_temp, mode=:pdf));
    end

    return h_normalized, h_unnormalized;
end
