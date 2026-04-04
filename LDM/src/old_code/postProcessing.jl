using LaTeXStrings
using StatsPlots
import NearestNeighbors
import PyPlot
#=
I'll assume that post-processing acts on a Dataset type.
=#

function plotLogPDFWithGaussian(arr::FPArray{1};
                                nbins=200,
                                label="data")
    arrCpy = arr ./ sqrt(mean(arr.^2));
    m = mean(arrCpy);
    s = std(arrCpy);    
    g(x) = (1/(s*sqrt(2π)))*exp(-0.5*(x-m)^2/(s^2)); #define gaussian using mean, variance
    edges = range(-6, 6, length=nbins);
    hist = fit(Histogram, arrCpy, edges);
    xs = midpoints(hist.edges[1]);
    dx = xs[2]-xs[1];
    goodInds = findall(x->x!=0.0, hist.weights);
    plt = plot(xs[goodInds], hist.weights[goodInds] / (sum(hist.weights)*dx), label=label);
    plot!(plt, xs, g.(xs), linestyle=:dash, label="Gaussian");
    plot!(plt, yaxis=:log);
    m_string = @sprintf(" %.2f; ", m);
    s_string = @sprintf(" %.2f; ", s);
    sk_string = @sprintf(" %.2f ", skewness(arrCpy));
    title = "μ =" * m_string * "σ =" * s_string * "skew = " * sk_string;
    plot!(plt, title=title)
    return plt;
end

function plotPDFOnDiagonal(vgt::FPArray{4})
    _,_,x,y = size(vgt);
    return plotPDFOnDiagonal(reshape(vgt, (3,3,x*y)));
end

function plotPDFOnDiagonal(vgt::FPArray{3})
    h1 = plotLogPDFWithGaussian(vgt[1,1,:]);
    plot!(h1, 
          legend=false, 
          title="A_11; mean=$(mean(vgt[1,1,:])), skew=$(skewness(vgt[1,1,:]))", titlefontsize=10,
          ylabel="Probability");
    h2 = plotLogPDFWithGaussian(vgt[2,2,:]);
    plot!(h2, 
          legend=false, 
          title="A_22; mean=$(mean(vgt[2,2,:])), skew=$(skewness(vgt[2,2,:]))", titlefontsize=10,
          ylabel="Probability");
    h3 = plotLogPDFWithGaussian(vgt[3,3,:]);
    plot!(h3, 
          legend=false, 
          title="A_33; mean=$(mean(vgt[3,3,:])), skew=$(skewness(vgt[3,3,:]))", titlefontsize=10,
          ylabel="Probability");
    return plot(h1,h2,h3, layout=(3,1), size=(800,800));
end

function plotPDFOffDiagonal(vgt::FPArray{4})
    _,_,x,y = size(vgt);
    return plotPDFOffDiagonal(reshape(vgt, (3,3,x*y)));
end

function plotPDFOffDiagonal(vgt::FPArray{3})
    h12 = plotLogPDFWithGaussian(vgt[1,2,:]);
    plot!(h12, 
          legend=false, 
          title="A_12; mean=$(mean(vgt[1,2,:])), skew=$(skewness(vgt[1,2,:]))", titlefontsize=10,
          ylabel="Probability");
    h23 = plotLogPDFWithGaussian(vgt[2,3,:]);
    plot!(h23, 
          legend=false, 
          title="A_23; mean=$(mean(vgt[2,3,:])), skew=$(skewness(vgt[2,3,:]))", titlefontsize=10,
          ylabel="Probability");
    h31 = plotLogPDFWithGaussian(vgt[3,1,:]);
    plot!(h31, 
          legend=false, 
          title="A_31; mean=$(mean(vgt[3,1,:])), skew=$(skewness(vgt[3,1,:]))", titlefontsize=10,
          ylabel="Probability");
    return plot(h12,h23,h31, layout=(3,1), size=(800,800));
end

function calculateQR(M::FPArray{3})
    _,_,numSamples = size(M);
    result = Dict("Q"=>zeros(numSamples),
                  "R"=>zeros(numSamples));
    
    Threads.@threads for i in 1:numSamples
        result["Q"][i], result["R"][i] = calculateQR(M[:,:,i]);
    end
    return result;
end

function calculateQR(M::FPArray{2})
    Q = (tr(M^2)/2.0);
    R = tr(M^3)/3.0;
    return Q,R;
end

function plotQRHeatmap(vgt::FPArray{3}; nbins=50)
    invars = calculateQR(vgt);
    xrange = -5.0:(10.0/nbins):5.0;
    yrange = -5.0:(10.0/nbins):5.0;
    hist = fit(Histogram,  (invars["R"], invars["Q"]), (xrange, yrange));
    p = heatmap(hist.edges[1], hist.edges[2], hist.weights/sum(hist.weights));
    plot!(p, xlabel="R*", ylabel="Q*", xlims=(-5,5), ylims=(-5,5));
    return p;
end

function calc_dQdR_Pressure(A::FPArray{2}, H::FPArray{2})
    intermed = A*A*H;
    dQ,dR = tr(intermed), tr(A*intermed);
    return dQ, dR;
end

function calc_dQdR_Pressure(vgt::FPArray{3}, ph::FPArray{3})
    @assert size(vgt) == size(ph);
    numSamples = size(vgt)[end];
    result = Dict("dQ"=>zeros(numSamples),
                  "dR"=>zeros(numSamples));
    for i in 1:numSamples
        result["dQ"][i], result["dR"][i] = calc_dQdR_Pressure(vgt[:,:,i], ph[:,:,i]);
    end
    return result;
end
        
function calc_rqFromSample(vgt::Matrix)
    @assert size(vgt) == (3,3);
    return r,q = -tr(vgt^3)/3.0, -tr(vgt^2)/2.0;
end

numTests = 10000;
# @testset "Q/R calculation" begin
#     for i in 1:numTests
#         sample = rand(1:numTrajs); 
#         M = aij[:,:,sample];
#         r,q = calc_rqFromSample(M);
#         py_q, py_r = py"getqr(aij[$(sample)-1,:,:].reshape([1,3,3]))";
#         @test q ≈ py_q[1];
#         @test r ≈ py_r[1];
#     end
# end

function calc_drdqFromPressure(vgt::Matrix, ph::Matrix)
    @assert size(vgt) == size(ph) == (3,3);
    return dr,dq = tr(vgt*vgt*ph), tr(vgt*ph);
end

function plotPressureQuiverComparison(vgt, ph1, ph2, outputPath; scaleFactor=0.1, labels=["", ""])
    @assert size(vgt) == size(ph1) == size(ph2);
    numSamples = size(vgt)[end];

    wnorm = sqrt(mean([norm((0.5*(vgt[:,:,i]-transpose(vgt[:,:,i])))^2) for i in 1:numSamples]));
    r = zeros(numSamples);
    q = zeros(numSamples);
    dr_1= zeros(numSamples);
    dq_1= zeros(numSamples);
    dr_2= zeros(numSamples);
    dq_2= zeros(numSamples);
    Threads.@threads for i in 1:numSamples
        r[i],q[i] = calc_rqFromSample(vgt[:,:,i]);
        dr_1[i],dq_1[i] = calc_drdqFromPressure(vgt[:,:,i],
                                                ph1[:,:,i]);
        dr_2[i],dq_2[i] = calc_drdqFromPressure(vgt[:,:,i],
                                                ph2[:,:,i]);
    end
    q  ./= wnorm^2;
    r  ./= wnorm^3;
    dq_1 ./= wnorm^3;
    dr_1 ./= wnorm^4;
    dq_2 ./= wnorm^3;
    dr_2 ./= wnorm^4;

    rRange = range(-5, 5, length=21);
    qRange = range(-5, 5, length=21);
    rMid = Array(midpoints(rRange));
    qMid = Array(midpoints(qRange));
    seperatix_r = range(rRange[1], rRange[end], length=1000);
    seperatix_q = -((27.0/4.0).*(seperatix_r.^2)).^(1/3);

    dq_array_1 = zeros(length(rMid), length(qMid));
    dr_array_1 = zeros(length(rMid), length(qMid));
    dq_array_2 = zeros(length(rMid), length(qMid));
    dr_array_2 = zeros(length(rMid), length(qMid));

    Threads.@threads for i in 1:length(rMid)
        ind_r = findall(x->rRange[i+1] > x > rRange[i], r);
        for j in 1:length(qMid)
            ind_q = findall(x->qRange[j+1] > x > qRange[j],q[ind_r]);
            if (length(ind_q) > 50)
                dq_array_1[i,j] = mean(dq_1[ind_r][ind_q]);
                dr_array_1[i,j] = mean(dr_1[ind_r][ind_q]);
                dq_array_2[i,j] = mean(dq_2[ind_r][ind_q]);
                dr_array_2[i,j] = mean(dr_2[ind_r][ind_q]);
            end
        end
    end

    quiverX = zeros(length(rMid)*length(qMid));
    quiverY = zeros(length(rMid)*length(qMid));
    quiverDX_1 = zeros(length(rMid)*length(qMid));
    quiverDY_1 = zeros(length(rMid)*length(qMid));
    quiverDX_2 = zeros(length(rMid)*length(qMid));
    quiverDY_2 = zeros(length(rMid)*length(qMid));
    for i in 1:length(rMid)
        for j in 1:length(qMid)
            ind = (i-1)*length(qMid) + j;
            quiverX[ind] = rMid[i];
            quiverY[ind] = qMid[j];
            quiverDX_1[ind] = dr_array_1[i,j];
            quiverDY_1[ind] = dq_array_1[i,j];
            quiverDX_2[ind] = dr_array_2[i,j];
            quiverDY_2[ind] = dq_array_2[i,j];
        end
    end

    quiverDX_1 .*= scaleFactor;
    quiverDY_1 .*= scaleFactor;
    quiverDX_2 .*= scaleFactor;
    quiverDY_2 .*= scaleFactor;

    py"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt;
    import numpy as np;
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.set_ylabel('Q*');
    ax.set_xlabel('R*');
    ax.set_title('Pressure Hessian contribution to Q-R CMTs');
    plt.plot($(seperatix_r), $(seperatix_q));
    plt.quiver($(quiverX),$(quiverY), $(quiverDX_1), $(quiverDY_1), color='b', label=$(labels[1]));
    plt.quiver($(quiverX),$(quiverY), $(quiverDX_2), $(quiverDY_2), color='r', label=$(labels[2]));
    plt.legend();
    plt.savefig($(outputPath));
    """
    return;
end

function plotPressureQuiver(vgt_array, ph_array, outputPath; scaleFactor=0.1, plotToOverlay=nothing, label="")
    @assert size(vgt_array) == size(ph_array);
    numSamples = size(vgt_array)[end];
    
    wnorm = sqrt(mean([norm((0.5*(vgt_array[:,:,i]-transpose(vgt_array[:,:,i])))^2) for i in 1:numSamples]));
    r = zeros(numSamples);
    q = zeros(numSamples);
    dr= zeros(numSamples);
    dq= zeros(numSamples);
    Threads.@threads for i in 1:numSamples
        r[i],q[i] = calc_rqFromSample(vgt_array[:,:,i]);
        dr[i],dq[i] = calc_drdqFromPressure(vgt_array[:,:,i],
                                            ph_array[:,:,i]);
    end
    q  ./= wnorm^2;
    r  ./= wnorm^3;
    dq ./= wnorm^3;
    dr ./= wnorm^4;

    rRange = range(-5, 5, length=21);
    qRange = range(-5, 5, length=21);
    rMid = Array(midpoints(rRange));
    qMid = Array(midpoints(qRange));
    seperatix_r = range(rRange[1], rRange[end], length=1000);
    seperatix_q = -((27.0/4.0).*(seperatix_r.^2)).^(1/3);

    dq_array = zeros(length(rMid), length(qMid));
    dr_array = zeros(length(rMid), length(qMid));

    Threads.@threads for i in 1:length(rMid)
        ind_r = findall(x->rRange[i+1] > x > rRange[i], r);
        for j in 1:length(qMid)
            ind_q = findall(x->qRange[j+1] > x > qRange[j],q[ind_r]);
            if (length(ind_q) > 20)
                dq_array[i,j] = mean(dq[ind_r][ind_q]);
                dr_array[i,j] = mean(dr[ind_r][ind_q]);
            end
        end
    end

    quiverX = zeros(length(rMid)*length(qMid));
    quiverY = zeros(length(rMid)*length(qMid));
    quiverDX = zeros(length(rMid)*length(qMid));
    quiverDY = zeros(length(rMid)*length(qMid));
    for i in 1:length(rMid)
        for j in 1:length(qMid)
            ind = (i-1)*length(qMid) + j;
            quiverX[ind] = rMid[i];
            quiverY[ind] = qMid[j];
            quiverDX[ind] = dr_array[i,j];
            quiverDY[ind] = dq_array[i,j];
        end
    end

    quiverDX .*= scaleFactor;
    quiverDY .*= scaleFactor;

    py"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt;
    import numpy as np;
    fig = plt.figure();
    ax = fig.add_subplot(111);
    ax.set_ylabel('Q*');
    ax.set_xlabel('R*');
    ax.set_title('Pressure Hessian contribution to Q-R CMTs');
    plt.plot($(seperatix_r), $(seperatix_q));
    plt.quiver($(quiverX),$(quiverY), $(quiverDX), $(quiverDY), color='r', label=$(label));
    plt.legend();
    plt.savefig($(outputPath));
    """
    return;
end

function calcAvgSkewness(vgt::FPArray{3})
    s1 = mean(vgt[1,1,:].^3)/(mean(vgt[1,1,:].^2)^(3/2));
    s2 = mean(vgt[2,2,:].^3)/(mean(vgt[2,2,:].^2)^(3/2));
    s3 = mean(vgt[3,3,:].^3)/(mean(vgt[3,3,:].^2)^(3/2));
    return (1/3)*(s1 + s2 + s3);
end

function plotDeviationFromPoisson(vgt::FPArray{3}, ph::FPArray{3}; bins=nothing)
    N = size(vgt)[end];
    vgtSum = zeros(N);
    for n in 1:N
        for i in 1:3
            for j in 1:3
                vgtSum[n] += vgt[i,j,n]*vgt[j,i,n];
            end
        end
    end

    dev = [tr(ph[:,:,n])+vgtSum[n] for n in 1:N];
    m_string = @sprintf(" %.4f; ", mean(dev));
    s_string = @sprintf(" %.4f; ", std(dev));
    title_string = "μ = " * m_string * "σ = " * s_string;
    if (bins != nothing)
        h = histogram(dev, bins=bins, xlabel="tr(H)+tr(A^2)", title=title_string, xaxis=:log);
    else
        h = histogram(dev, xlabel="tr(H)+tr(A^2)", title=title_string, xaxis=:log);
    end
    return h;
end

function plotPredictedTBWeightDistributions(vgt::FPArray{3}, nn, params;
                                            nbins = 100)
    numSamples = size(vgt)[end];
    invars = zeros(numInvariants, numSamples);
    calcInvariants!(invars, vgt);
    output = nn(params)(invars);

    plt_g1 = histogram(output[1,:], nbins=nbins, label=L"\delta" * " predicted");
    plt_g2 = histogram(output[2,:], nbins=nbins, label=L"\gamma" * " predicted");
    plt_g3 = histogram(output[3,:], nbins=nbins, label=L"\beta" *  " predicted");
    plt_g4 = histogram(output[4,:], nbins=nbins, label=L"\alpha" * " predicted");

    alpha = -0.75;
    beta  = -0.67;
    gamma = 0.15;
    delta = -0.11;
    
    scatter!(plt_g4, [alpha], [0], label="Lawson & Dawson");
    scatter!(plt_g3, [beta], [0], label="Lawson & Dawson");
    scatter!(plt_g2, [gamma], [0], label="Lawson & Dawson");
    scatter!(plt_g1, [delta], [0], label="Lawson & Dawson");

    p = plot(plt_g1, plt_g2, plt_g3, plt_g4, layout=(2,2));
    return p;
end

function calcHistogramEigenvectorAlignment(predPH, gtPH;
                                           nbins = 40)
    @assert size(predPH) == size(gtPH);
    numSamples = size(predPH)[end];
    alpha_theta = zeros(numSamples);
    beta_theta = zeros(numSamples);
    gamma_theta = zeros(numSamples);

    clip(val) = max(-1,min(1,val));

    Threads.@threads for i in 1:numSamples
        predVecs = eigen(predPH[:,:,i]).vectors;
        gtVecs =   eigen(gtPH[:,:,i]).vectors;
        alpha_theta[i] = acos(clip(dot(predVecs[:,3], gtVecs[:,3])));
        beta_theta[i]  = acos(clip(dot(predVecs[:,2], gtVecs[:,2])));
        gamma_theta[i] = acos(clip(dot(predVecs[:,1], gtVecs[:,1])));
    end
    
    xlims = (0.0, π/2);
    edges = range(xlims[1], stop=xlims[2], length=nbins);
    h_alpha = fit(Histogram, alpha_theta, edges);
    h_beta = fit(Histogram, beta_theta, edges);
    h_gamma = fit(Histogram, gamma_theta, edges);

    h_alpha = normalize(h_alpha, mode=:pdf);
    h_beta  = normalize(h_beta, mode=:pdf);
    h_gamma = normalize(h_gamma, mode=:pdf);
    
    return [h_alpha, h_beta, h_gamma];
end

function plotPressureEigenvectorAlignment(predPH, gtPH; nbins=20)
    h_alpha, h_beta, h_gamma = calcHistogramEigenvectorAlignment(predPH, gtPH, nbins);
    xs = midpoints(h_alpha.edges[1]);
    
    p_alpha = plot(xs, h_alpha.weights,
                   xlabel = L"acos(\vec{p_{pred,\alpha}} \cdot \vec{p_{gt,\alpha}})");
    p_beta = plot(xs, h_beta.weights,
                   xlabel = L"acos(\vec{p_{pred,\beta}} \cdot \vec{p_{gt,\beta}})");
    p_gamma = plot(xs, h_gamma.weights,
                   xlabel = L"acos(\vec{p_{pred,\gamma}} \cdot \vec{p_{gt,\gamma}})");

    return plot(p_alpha, p_beta, p_gamma,
                layout=(1,3),
                size=(1000,300),
                plot_title="Pressure Hessian Eigenvector Alignment PDFs: " * L"\alpha > \gamma > \beta",
                margin=6Plots.mm);
end

function plotLoss(lossArray, gtPH)
    # numEpochs = length(lossArray);

    # flattenedPH = reshape(gtPH, (9,size(gtPH)[end]));
    # noModelLoss = Flux.Losses.mse(zeros(size(flattenedPH)), flattenedPH);
    # todo, this is a nice idea, but I need normalization as well...natural place
    # for this calculation is inside the algorithm
    p = plot(lossArray[1,:], yaxis=:log, label="MSE(H-H_pred)_train");
    plot!(p, lossArray[2,:], yaxis=:log, label="MSE(H-H_pred)_test");
    plot!(p, xlabel="epochs", ylabel="Loss", title="Loss vs Epoch");
    # plot!(p, [noModelLoss for i in 1:numEpochs], label="MSE(H-0)");

    return p;
end

function plotPressureStrainEigPDF(ph::FPArray{3}, vgt::FPArray{3})
    @assert size(ph) == size(vgt);
    numSamples = size(ph)[end];

    s1p1 = zeros(numSamples);
    s1p2 = zeros(numSamples);
    s1p3 = zeros(numSamples);

    s2p1 = zeros(numSamples);
    s2p2 = zeros(numSamples);
    s2p3 = zeros(numSamples);

    s3p1 = zeros(numSamples);
    s3p2 = zeros(numSamples);
    s3p3 = zeros(numSamples);

    Threads.@threads for i in 1:numSamples
        S = 0.5*(vgt[:,:,i] + vgt[:,:,i]');
        sVecs = eigen(S).vectors;
        pVecs = eigen(ph[:,:,i]).vectors;

        s1p1[i] = dot(sVecs[:,1], pVecs[:,1]);
        s1p2[i] = dot(sVecs[:,1], pVecs[:,2]);
        s1p3[i] = dot(sVecs[:,1], pVecs[:,3]);

        s2p1[i] = dot(sVecs[:,2], pVecs[:,1]);
        s2p2[i] = dot(sVecs[:,2], pVecs[:,2]);
        s2p3[i] = dot(sVecs[:,2], pVecs[:,3]);

        s3p1[i] = dot(sVecs[:,3], pVecs[:,1]);
        s3p2[i] = dot(sVecs[:,3], pVecs[:,2]);
        s3p3[i] = dot(sVecs[:,3], pVecs[:,3]);
    end

    h11 = fit(Histogram, s1p1);
    h12 = fit(Histogram, s1p2);
    h13 = fit(Histogram, s1p3);
    h11 = normalize(h11);
    h12 = normalize(h12);
    h13 = normalize(h13);

    h21 = fit(Histogram, s2p1);
    h22 = fit(Histogram, s2p2);
    h23 = fit(Histogram, s2p3);
    h21 = normalize(h21);
    h22 = normalize(h22);
    h23 = normalize(h23);

    h31 = fit(Histogram, s3p1);
    h32 = fit(Histogram, s3p2);
    h33 = fit(Histogram, s3p3);
    h31 = normalize(h31);
    h32 = normalize(h32);
    h33 = normalize(h33);

    plt1 = plot(midpoints(h11.edges[1]), h11.weights, label = L"s_{\gamma} \cdot p_{\gamma}");
    plot!(plt1, midpoints(h12.edges[1]), h12.weights, label = L"s_{\gamma} \cdot p_{\beta}");
    plot!(plt1, midpoints(h13.edges[1]), h13.weights, label = L"s_{\gamma} \cdot p_{\alpha}");
    plot!(plt1, xlabel= L"s_{\gamma} \cdot p_i", margin=6Plots.mm);
    plot!(plt1, ylims=(0.0,2.0), xlims=(-1.0,1.0));

    plt2 = plot(midpoints(h21.edges[1]), h21.weights, label = L"s_{\beta} \cdot p_{\gamma}");
    plot!(plt2, midpoints(h22.edges[1]), h22.weights, label = L"s_{\beta} \cdot p_{\beta}");
    plot!(plt2, midpoints(h23.edges[1]), h23.weights, label = L"s_{\beta} \cdot p_{\alpha}");
    plot!(plt2, xlabel= L"s_{\beta} \cdot p_i", margin=6Plots.mm);
    plot!(plt2, ylims=(0.0,2.0), xlims=(-1.0,1.0));

    plt3 = plot(midpoints(h31.edges[1]), h31.weights, label = L"s_{\alpha} \cdot p_{\gamma}");
    plot!(plt3, midpoints(h32.edges[1]), h32.weights, label = L"s_{\alpha} \cdot p_{\beta}"); 
    plot!(plt3, midpoints(h33.edges[1]), h33.weights, label = L"s_{\alpha} \cdot p_{\alpha}");
    plot!(plt3, xlabel= L"s_{\alpha} \cdot p_i", margin=6Plots.mm);
    plot!(plt3, ylims=(0.0,2.0), xlims=(-1.0,1.0));

    return plot(plt3, plt2, plt1,
                layout = (1,3),
                size=(1000,300),
                plot_title="Eigenvector Alignment of: " * L"\hat{H}, \hat{S} \quad \alpha > \beta > \gamma");
end

function plotPressureEigenvaluePDF(predPH::FPArray{3},
                                   gtPH::FPArray{3};
                                   nbins=25)
    @assert size(predPH) == size(gtPH);
    numSamples = size(predPH)[end];

    alpha_gt = zeros(numSamples);
    beta_gt =  zeros(numSamples);
    gamma_gt = zeros(numSamples);

    alpha_pred = zeros(numSamples);
    beta_pred =  zeros(numSamples);
    gamma_pred = zeros(numSamples);
    
    Threads.@threads for i in 1:numSamples
        gamma_gt[i], beta_gt[i], alpha_gt[i] = eigen(gtPH[:,:,i]).values;
        gamma_pred[i], beta_pred[i], alpha_pred[i] = eigen(predPH[:,:,i]).values;
    end

    h_alpha_gt = fit(Histogram, alpha_gt./std(alpha_gt), nbins=nbins);
    h_alpha_gt.weights .+= 1;
    h_alpha_gt = normalize(h_alpha_gt, mode=:pdf);

    h_beta_gt = fit(Histogram, beta_gt./std(beta_gt), nbins=nbins);
    h_beta_gt.weights .+= 1;
    h_beta_gt = normalize(h_beta_gt, mode=:pdf);

    h_gamma_gt = fit(Histogram, gamma_gt./std(gamma_gt), nbins=nbins);
    h_gamma_gt.weights .+= 1;
    h_gamma_gt = normalize(h_gamma_gt, mode=:pdf);

    h_alpha_pred = fit(Histogram, alpha_pred./std(alpha_pred), nbins=nbins);
    h_alpha_pred.weights .+= 1;
    h_alpha_pred = normalize(h_alpha_pred, mode=:pdf);

    h_beta_pred = fit(Histogram, beta_pred./std(beta_pred), nbins=nbins);
    h_beta_pred.weights .+= 1;
    h_beta_pred = normalize(h_beta_pred, mode=:pdf);

    h_gamma_pred = fit(Histogram, gamma_pred./std(gamma_pred), nbins=nbins);
    h_gamma_pred.weights .+= 1;
    h_gamma_pred = normalize(h_gamma_pred, mode=:pdf);

    plt_alpha = plot(midpoints(h_alpha_gt.edges[1]), h_alpha_gt.weights, label="gt");
    plot!(plt_alpha, midpoints(h_alpha_pred.edges[1]), h_alpha_pred.weights, label="pred");
    plot!(plt_alpha,
          xlims = (h_alpha_gt.edges[1][1], h_alpha_gt.edges[1][end]),
          xlabel = L"p_\alpha",
          ylims = (1e-4, 1e0),
          yaxis=:log);

    plt_beta = plot(midpoints(h_beta_gt.edges[1]), h_beta_gt.weights, label="gt");
    plot!(plt_beta, midpoints(h_beta_pred.edges[1]), h_beta_pred.weights, label="pred");
    plot!(plt_beta,
          xlims = (h_beta_gt.edges[1][1], h_beta_gt.edges[1][end]),
          xlabel = L"p_\beta",
          ylims = (1e-4, 1e0),
          yaxis=:log);

    plt_gamma = plot(midpoints(h_gamma_gt.edges[1]), h_gamma_gt.weights, label="gt");
    plot!(plt_gamma, midpoints(h_gamma_pred.edges[1]), h_gamma_pred.weights, label="pred");
    plot!(plt_gamma,
          xlims = (h_gamma_gt.edges[1][1], h_gamma_gt.edges[1][end]),
          xlabel = L"p_\gamma",
          ylims = (1e-4, 1e0),
          legend = :topleft,
          yaxis=:log);

    return plot(plt_alpha, plt_beta, plt_gamma,
                plot_title = "PDF of eigenvalues of PH: " * L"\alpha > \beta > \gamma",
                ylabel = "PDF",
                layout = (1, 3),
                size=(1000,300),
                margin=6Plots.mm);
end

function plotSStarPDF(predPH::FPArray{3}, gtPH::FPArray{3})

    @assert size(predPH) == size(gtPH);
    numSamples = size(predPH)[end];
    
    function calcSStar(H::FPArray{2})
        return -sqrt(6)*(tr(H^3)/(tr(H^2)^(3/2)));
    end
    predSStar = [calcSStar(predPH[:,:,i]) for i in 1:numSamples];
    gtSStar = [calcSStar(gtPH[:,:,i]) for i in 1:numSamples];

    edges = range(-1, 1, length=21);
    pred_h = fit(Histogram, predSStar, edges)
    pred_h = normalize(pred_h, mode=:pdf);

    gt_h = fit(Histogram, gtSStar, edges);
    gt_h = normalize(gt_h, mode=:pdf);

    plt = plot(midpoints(edges), pred_h.weights, label="predicted");
    plot!(plt, midpoints(edges),   gt_h.weights,   label="ground truth");
    plot!(plt, xlabel = "\$ s^* \$", margin=6Plots.mm);
    plot!(plt, ylabel = "PDF");
    plot!(plt, legend = :topleft);
    plot!(plt, xlims=(edges[1], edges[end]));
    plot!(plt, ylims=(0.3, 1.0));
    plot!(plt, title=L"s^* = -\frac{3\sqrt{6} \ \phi_1 \phi_2 \phi_3}{(\phi_1^2 + \phi_2^2 + \phi_3^2)^{3/2}}")
end

function plotLumleyTriangle(ph::FPArray{3})
    numSamples = size(ph)[end];
    
    ζ = zeros(numSamples);
    x = zeros(numSamples);

    Threads.@threads for i in 1:numSamples
        b = ph[:,:,i]/norm(ph[:,:,i]);
        ζ[i] = -sqrt(6)*tr(b^3);
        x[i] = (tr(b^2))^(3/2);
    end

    h = fit(Histogram, (ζ, x));

    return plot(h, xlabel="\$ \\zeta \$", ylabel="\$ \\chi \$");
end

function postProcessParashar(vgt::FPArray{3},
                             gtPH::FPArray{3},
                             outputPath::String,
                             Fs,
                             Gs)
    #todo, this is ambiguous...it should work for the moment because we're including the file
    # at the 'right' place, but we shouldn't rely on "predictPHFromVGT" to be defined
    # by some unknown module before our use here.
    predPH = predictPHFromVGT(vgt, Fs, Gs);

    postProcessCommon(vgt,
                      gtPH,
                      predPH,
                      outputPath,
                      ones(1),
                      zeros(1));
end

function postProcessCommon(vgt::FPArray{3},
                           gtPH::FPArray{3},
                           predPH::FPArray{3},
                           outputPath::String,
                           lossArray::FPArray{2},
                           ps)
    
    mkpath(outputPath);
    serialize(outputPath * "/loss.jls", lossArray);
    serialize(outputPath * "/params.jls", ps);
    serialize(outputPath * "/eigAlignHists.jls",
              calcHistogramEigenvectorAlignment(predPH, gtPH));

    figPath = outputPath * "/figures/";
    mkpath(figPath);
    plotPressureQuiverComparison(vgt, gtPH, predPH, figPath * "comp_qrCMT.png", labels=["ground truth", "predicted"]);

    p_pred = plotPressureStrainEigPDF(predPH, vgt);
    p_gt = plotPressureStrainEigPDF(gtPH, vgt);
    savefig(plot(p_gt,p_pred,layout=(2,1), size=(1000,600)), figPath * "pressureStrainAlign.png");
    savefig(plotSStarPDF(predPH, gtPH), figPath * "pressureShapeParamPDF.png");

    # savefig(plot(plotLumleyTriangle(predPH),
    #              plotLumleyTriangle(gtPH),
    #              layout = (1,2), size=(1000, 500)), figPath * "lumleyTriangle.png");

    savefig(plotPressureEigenvectorAlignment(predPH, gtPH; nbins=20),
            figPath * "pressureEigenvalueAlignment.png");

    savefig(plotPressureEigenvaluePDF(predPH,gtPH),
            figPath * "pressureEigenvaluePDF.png");
end

function postProcessTBNN(vgt::FPArray{3},
                         gtPH::FPArray{3},
                         outputPath::String,
                         lossArray::FPArray{2},
                         params,
                         nn)
    #todo, this is ambiguous...it should work for the moment because we're including the file
    # at the 'right' place, but we shouldn't rely on "predictPHFromVGT" to be defined
    # by some unknown module before our use here.
    predPH = predictPHFromVGT(vgt, nn, params); 

    postProcessCommon(vgt, gtPH, predPH, outputPath, lossArray, params);
    savefig(plotPredictedTBWeightDistributions(vgt, nn, params),
            outputPath * "/figures/pred_distTBWeights.png");
end

function postProcessFF(vgt::FPArray{3},
                       gtPH::FPArray{3},
                       outputPath::String,
                       lossArray::FPArray{2},
                       params,
                       nn)
    #todo, this is ambiguous...it should work for the moment because we're including the file
    # at the 'right' place, but we shouldn't rely on "predictPHFromVGT" to be defined
    # by some unknown module before our use here.
    predPH = predictPHFromVGT(vgt, nn, params); 

    postProcessCommon(vgt, gtPH, predPH, outputPath, lossArray, params);
end

function plotInvariantsGroupedBoxPlot(invariants::FPArray{2})
    numInvariants, numSamples = size(invariants);
    xs = zeros(Int, size(invariants));
    groups = zeros(Int, size(invariants));

    for i in 1:numInvariants
        for j in 1:numSamples
            xs[i,j] = i;
            groups[i,j] = i;
        end
    end

    linInvar = reshape(invariants, numInvariants*numSamples);
    linXs = reshape(xs, numInvariants*numSamples);
    linGroups = reshape(groups, numInvariants*numSamples);

    return groupedboxplot(linXs, linInvar, group=linGroups, outliers=false);
end

function plotInvariantsBoxPlot(invariants::FPArray{2})
    numInvariants, numSamples = size(invariants);
    xs = zeros(Int, size(invariants));
    groups = zeros(Int, size(invariants));

    for i in 1:numInvariants
        for j in 1:numSamples
            xs[i,j] = i;
            groups[i,j] = i;
        end
    end

    linInvar = reshape(invariants, numInvariants*numSamples);
    linXs = reshape(xs, numInvariants*numSamples);
    linGroups = reshape(groups, numInvariants*numSamples);

    return boxplot(linXs, linInvar, outliers=false);
end

function plotInvariantsBoxPlot_string(invariants::FPArray{2})
    numInvariants, numSamples = size(invariants);
    xs = fill("", size(invariants));

    for i in 1:numInvariants
        for j in 1:numSamples
            st = "λ_$(i)"
            xs[i,j] = st;
        end
    end

    linInvar = reshape(invariants, numInvariants*numSamples);
    linXs = reshape(xs, numInvariants*numSamples);

    return boxplot(linXs, linInvar, outliers=false);
end

function calcScalarsAndTBContributions(pathsToParams::Array{String, 1},
                                       pathToData::String,
                                       outputDir::String;
                                       hiddenDim = 30)

    p, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, numBasisElements)));

    data = LagrangianDataset(deserialize(pathToData));
    N = 262144;
    data = LagrangianDataset(data.vgt[:,:,:,1:N],
                             data.ph[:,:,:,1:N],
                             data.vis,
                             data.sgs,
                             data.pathToSource,
                             data.regenerate,
                             data.filtersize,
                             data.dt);

    means = zeros(numBasisElements, length(pathsToParams));
    vars = zeros(numBasisElements, length(pathsToParams));
    outputs = zeros(numBasisElements, N, length(pathsToParams));
    basisContributions = zeros(numBasisElements, N, length(pathsToParams));
    vgt = data.vgt[:,:,1,:];
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    normalizedInvars = calcInvariants(normalizedVGT);
    normalizedTB = calcSymmetricTensorBasis(normalizedVGT);

    i = 1;
    for path in pathsToParams
        params = deserialize(path);
        scalars = nn(params)(normalizedInvars); #(numBasisElements,N)
        outputs[:,:,i] .= scalars;
        for j in 1:numBasisElements
            for n in 1:N
                basisContributions[j,n,i] = norm(scalars[j,n]*normalizedTB[j,n]);
            end
        end
        # means[:,i] .= mean(scalars, dims=2);
        # vars[:,i] .= var(scalars, dims=2);
        i += 1;
    end

    # serialize(outputDir * "/scalarFunctionMeans.jls", means);
    # serialize(outputDir * "/scalarFunctionVars.jls",  vars);
    serialize(outputDir * "/scalarFunctionVals.jls", outputs);
    serialize(outputDir * "/basisContributions.jls", basisContributions);
end

function plotSensitivityOfGToInvariants(pathToParams,
                                        pathToData;
                                        N = 262144,
                                        hiddenDim = 30)

    _, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, numBasisElements)));

    data = LagrangianDataset(deserialize(pathToData));
    vgt = data.vgt[:,:,1,1:N];
    params = deserialize(pathToParams);

    p = plotSensitivityOfGToInvariants(nn, params, vgt);

    return p;
end

function plotPercentContributions(pathToParams, pathToData; hiddenDim=30, N=262144)
    _, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, numBasisElements)));

    data = LagrangianDataset(deserialize(pathToData));
    vgt = data.vgt[:,:,1,1:N];
    params = deserialize(pathToParams);

    vgt = data.vgt[:,:,1,:];
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    normalizedInvars = calcInvariants(normalizedVGT);
    normalizedTB = calcSymmetricTensorBasis(normalizedVGT);

    params = deserialize(pathToParams);
    scalars = nn(params)(normalizedInvars); #(numBasisElements,N)
    basisContributions = zeros(numBasisElements, N);

    Threads.@threads for n in 1:N
        tmp = [scalars[i,n]*normalizedTB[i,n] for i in 1:numBasisElements];
        H = sum(tmp);
        basisContributions[:,n] .= (norm.(tmp)) ./ norm(H);
    end
    
    p = boxplot(basisContributions', outliers=false, legend=false,
                ylabel = "\$ ||g^{(i)}T^{(i)}||_2 / ||H_{pred}||_2  \$",
                xticks = 1:1:10,
                title = "Magnitude of contributions of TBNN Summands");

    return p;
end

function plotSensitivityOfGToInvariants(nn, params, vgt::FPArray{3})
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    normalizedInvars = calcInvariants(normalizedVGT);
    unperturbedGs = nn(params)(normalizedInvars);
    plots = [];

    for i in 1:numInvariants
        tmpInvars = zeros(size(normalizedInvars));
        tmpInvars .= normalizedInvars;
        tmpInvars[i,:] .= mean(normalizedInvars[i,:]);
        perturbedGs = nn(params)(tmpInvars);
        dg = unperturbedGs - perturbedGs
        p = boxplot(dg', outliers=false, legend=false);
        plot!(p, xticks=([1,2,3,4,5,6,7,8,9,10], ["\$g^{(1)}\$",
                                                  "\$g^{(2)}\$",
                                                  "\$g^{(3)}\$",
                                                  "\$g^{(4)}\$",
                                                  "\$g^{(5)}\$",
                                                  "\$g^{(6)}\$",
                                                  "\$g^{(7)}\$",
                                                  "\$g^{(8)}\$",
                                                  "\$g^{(9)}\$",
                                                  "\$g^{(10)}\$"]));
        plot!(p, ylabel="\$ \\Delta g \$");
        plot!(p, ylims = (-0.5, 0.5));
        plot!(p, title = "\$ \\Delta g \$ for \$\\lambda_{$(i)} = \\mu(\\lambda_{$(i)})\$");
        push!(plots, p);
    end

    return plots;
end

function scatterMeansOfScalarVals(meanScalars::FPArray{2}; title=nothing, ylims=nothing, yaxis=nothing) #(numBasisElements, numTrials)
    numBasisElements, numTrials = size(meanScalars);
    xs = 1:numBasisElements;
    p = plot();
    for i in 1:numTrials
        for j in 1:numBasisElements
            scatter!(p, [xs[j]], [meanScalars[j,i]])
        end
    end
    plot!(p, legend=false)
    plot!(p, xticks=([1,2,3,4,5,6,7,8,9,10], [LaTeXString("g_(1)"),
                                              LaTeXString("g_(2)"),
                                              LaTeXString("g_(3)"),
                                              LaTeXString("g_(4)"),
                                              LaTeXString("g_(5)"),
                                              LaTeXString("g_(6)"),
                                              LaTeXString("g_(7)"),
                                              LaTeXString("g_(8)"),
                                              LaTeXString("g_(9)"),
                                              LaTeXString("g_(10)")]));
    plot!(p, title=title, ylims=ylims, yaxis=yaxis)
    return p;
end

function scatterMeansOfContributionVals(meanScalars::FPArray{2}; title=nothing, ylims=nothing, yaxis=nothing) #(numBasisElements, numTrials)
    numBasisElements, numTrials = size(meanScalars);
    xs = 1:numBasisElements;
    p = plot();
    for i in 1:numTrials
        for j in 1:numBasisElements
            scatter!(p, [xs[j]], [meanScalars[j,i]])
        end
    end
    plot!(p, legend=false)
    plot!(p, xticks=([1,2,3,4,5,6,7,8,9,10], [LaTeXString("g1*T1"),
                                              LaTeXString("g2*T2"),
                                              LaTeXString("g3*T3"),
                                              LaTeXString("g4*T4"),
                                              LaTeXString("g5*T5"),
                                              LaTeXString("g6*T6"),
                                              LaTeXString("g7*T7"),
                                              LaTeXString("g8*T8"),
                                              LaTeXString("g9*T9"),
                                              LaTeXString("g10*T10")]));
    plot!(p, title=title, ylims=ylims, yaxis=yaxis)
    return p;
end

function plotGFunctions(pathToParams, pathToData; hiddenDim=30, N=262144, gifs=false)
    _, nn = Flux.destructure(Chain(Dense(numInvariants,hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, hiddenDim, tanh),
                                   Dense(hiddenDim, numBasisElements)));

    data = LagrangianDataset(deserialize(pathToData));
    vgt = data.vgt[:,:,1,1:N];
    params = deserialize(pathToParams);

    p = plotGFunctions(nn, params, vgt, N=N, gifs=gifs);

    return p;
end

function plotGFunctions(nn, params, vgt; N=262144, numBins=50, gifs=false, k=5)
    timescale = calcCharacteristicTimescale(vgt);
    normalizedVGT = vgt * timescale;
    normalizedInvars = calcInvariants(normalizedVGT);
    gs = nn(params)(normalizedInvars);
    plots = [];

    tree = NearestNeighbors.BruteTree(normalizedInvars[1:2,1:N]);
    function interpolateData(x,y, i)
        idxs, dist = NearestNeighbors.knn(tree, [x; y], k);
        return mean(gs[i,idxs]);
    end

    for i in 1:numBasisElements
        xmin = 0.0;#minimum(normalizedInvars[1,:]);
        xmax = 0.25;#maximum(normalizedInvars[1,:]);
        ymin = -1.0;#minimum(normalizedInvars[2,:]);
        ymax = 0.0;#maximum(normalizedInvars[2,:]);
        xsteps = xmin:(xmax-xmin)/numBins:xmax;
        ysteps = ymin:(ymax-ymin)/numBins:ymax;
        s = surface(xsteps, ysteps, (x,y)->interpolateData(x,y,i));
        plot!(s, camera=(30,30));
        plot!(s, xlabel="\$\\lambda_1\$", ylabel="\$\\lambda_2\$", zlabel="\$g^{($(i))}(\\lambda_1, \\lambda_2)\$");
        push!(plots, s);

        if (gifs == true)
            anim = @animate for θ in 0:3:360
                plot!(plots[i], camera=(θ,30));
            end
            gif(anim, "./anim$(i)_fps15.gif", fps=15, loop=3);
        end
                
    end

    p = scatter(normalizedInvars[1,:], normalizedInvars[2,:], xlabel="\$\\lambda_1\$", ylabel="\$\\lambda_2\$", title="Locations of samples");
    push!(plots,p);

    return plots;
end
