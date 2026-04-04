const numInvariants = 5;
const numBasisElements = 10;

struct TBInputData
    invariants;
    tb;
end
struct TBLabeledData
    input::TBInputData;
    labels;
end

getInputData(data::TBLabeledData) = vcat(data.invariants,
                                         data.tb);

function calcSymmetricTensorBasis!(dest,
                                   mij)
    numSamples = size(dest)[end];
    
    Threads.@threads for i in 1:numSamples
        id = Matrix(I,3,3);
        S = 0.5*(mij[:,:,i]+mij[:,:,i]');
        W = 0.5*(mij[:,:,i]-mij[:,:,i]');

        dest[1,i]  = S # TB#1
        dest[2,i]  = S*W-W*S # TB#2
        dest[3,i]  = S^2 - (tr(S^2)/3)*id; # TB#3         
        dest[4,i]  = W^2 - (tr(W^2)/3)*id; # TB#4
        dest[5,i]  = W*S^2 - S^2*W # TB#5
        dest[6,i]  = W^2*S + S*W^2 - (2*tr(S*W^2)/3)*id; # TB#6
        dest[7,i]  = W*S*W^2 - W^2*S*W # TB#7
        dest[8,i]  = S*W*S^2 - S^2*W*S # TB#8
        dest[9,i]  = W^2*S^2 + S^2*W^2 - (2*tr(S^2*W^2)/3)*id; # TB#9
        dest[10,i] = W*S^2*W^2 - W^2*S^2*W  # TB#10
    end

    return;
end

function calcSymmetricTensorBasis(mij::FPArray{3})
    numSamples = size(mij)[end];
    result = [zeros(3,3) for i in 1:(numSamples*numBasisElements)];
    result = reshape(result, (numBasisElements, numSamples));
    calcSymmetricTensorBasis!(result, mij);
    return result;
end

function calcSymmetricTensorBasis(mij::FPArray{2})
    @assert size(mij) == (3,3);
    return calcSymmetricTensorBasis(reshape(mij, (3,3,1)));
end

function calcInvariants!(dest::FPArray{2},
                         mij::FPArray{3})
    numSamples = size(dest)[end];
    
    Threads.@threads for i in 1:numSamples
        S = 0.5*(mij[:,:,i]+mij[:,:,i]');
        W = 0.5*(mij[:,:,i]-mij[:,:,i]');
        dest[1,i]  = tr(S^2)      #λ_1
        dest[2,i]  = tr(W^2)      #λ_2
        dest[3,i]  = tr(S^3);     #λ_3
        dest[4,i] = tr(W^2 * S)   #λ_4
        dest[5,i] = tr(W^2 * S^2) #λ_5
    end
end

function calcInvariants(mij::FPArray{3})
    numSamples = size(mij)[end];
    result = zeros(numInvariants, numSamples);
    calcInvariants!(result, mij);
    return result;
end

function calcInvariants(vgt::FPArray{2})
    @assert size(vgt) == (3,3);
    return calcInvariants(reshape(vgt, (3,3,1)))[:,1];
end

function calcCharacteristicTimescale(vgt::FPArray{3})
    numSamples = size(vgt)[end];
    normArr = zeros(numSamples);
    Threads.@threads for i in 1:numSamples
        S = 0.5*(vgt[:,:,i]+vgt[:,:,i]');
        normArr[i] = norm(S^2);
    end
    timescale = 1/mean(normArr);
    return timescale;
end
