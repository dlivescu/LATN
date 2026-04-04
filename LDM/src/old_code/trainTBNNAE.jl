inputPath, outputPath, maxiters, latentDim = ARGS;
maxiters = parse(Int, maxiters);
latentDim = parse(Int, latentDim);

basepath = "/home/u1/cmhyett/LDM/";
using Pkg;
Pkg.activate(basepath);
Pkg.instantiate();
using LDM, Flux;
numInvariants = 5;
numBasisElements = 10;
hiddenDim = 30;

p, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                               Dense(hiddenDim, hiddenDim, tanh),
                               Dense(hiddenDim, latentDim, tanh),
                               Dense(hiddenDim, hiddenDim, tanh),
                               Dense(hiddenDim, hiddenDim, tanh),
                               Dense(hiddenDim, hiddenDim, tanh),
                               Dense(hiddenDim, hiddenDim, tanh),
                               Dense(hiddenDim, numBasisElements)));

LDM.TBNN.runTBNN(inputPath, outputPath, maxiters=maxiters, nn=nn, p=p);
