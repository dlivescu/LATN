using Flux, Plots, Serialization, Statistics;

datapath = "./results/lanl240/";

vgt = deserialize(datapath * "lanl_240_vgt.jls");
ph = deserialize(datapath * "lanl_240_ph.jls");
tau = LDM.TBNN.calcCharacteristicTimescale(vgt);

function robustScalar!(x)
    numDims,numSamples = size(x);
    for i in 1:numDims
        Q1,Q3 = quantile(x[i,:], [0.25, 0.75]);
        x[i,:] .= (x[i,:] .- Q1) ./ (Q3 - Q1);
    end
end

invars = LDM.TBNN.calcInvariants(vgt);
robustScalar!(invars);
N = floor(Int, 3*size(invars)[end]/4);
trainInvars = invars[:,1:N];
testInvars = invars[:,N+1:end];
dl = Flux.Data.DataLoader(trainInvars, batchsize = floor(Int,size(trainInvars)[end]/5), shuffle=true);
device = cpu;

function train!(l, p, opt, loader; numEpochs=10)
    lossArray = zeros(numEpochs);
    testLossArray = zeros(numEpochs);
    for epoch in 1:numEpochs
        for x in loader
            loss, back = Flux.pullback(p) do
                l(x |> device)
            end
            grad = back(1f0);
            Flux.Optimise.update!(opt, p, grad);
            lossArray[epoch] += loss;
        end
        lossArray[epoch] /= length(loader);
        testLossArray[epoch] = l(testInvars);
        if ((epoch-1) % 100 == 0)
            println("Loss = $(lossArray[epoch])");
        end
    end
    return lossArray, testLossArray;
end

numEpochs = 5000;
opt = ADAM();

p1 = plot();
p2 = plot();
for latentDim in 1:4
    model = Chain(Dense(5,4,leakyrelu),
                  Dense(4,latentDim,leakyrelu),
                  Dense(latentDim,4,leakyrelu),
                  Dense(4,5))
    ps = Flux.params(model);
    means = mean(invars, dims=2);
    stds = std(invars, dims=2);
    function loss(x)
        return Flux.Losses.mse(model(x),x);
        # pred = model(x);
        # diff = (x .- pred) ./ vs;
        # return mean(diff.^2);
    end

    lossArray,testLoss = train!(loss, ps, opt, dl, numEpochs = numEpochs);
    plot!(p1, lossArray, yaxis=:log, label="\$ d=$(latentDim)\$", color=latentDim);
    testInds = [i for i in 1:50:numEpochs];
    scatter!(p1, testInds, testLoss[testInds], yaxis=:log, label="", markershape=:xcross, color=latentDim);
    scatter!(p2, [latentDim], [minimum(lossArray)], yaxis=:log);
end
plot!(p1, ylabel="\$ MSE(NN(x),x)/MSE(0,x) \$", xlabel="epoch");
plot!(p2, ylabel="\$ \\min MSE(NN(x),x)/MSE(0,x) \$", xlabel="latent dimension");
plot!(p1);
