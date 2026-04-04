inputPath, outputPath, maxiters = ARGS;
maxiters = parse(Int, maxiters);

basepath = "./";
using Pkg;
Pkg.activate(basepath);
Pkg.instantiate();
using LDM;

LDM.TBNN.runTBNN(inputPath, outputPath, maxiters=maxiters);

