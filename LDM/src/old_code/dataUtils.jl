
"""
    unpackLagrData(basepath;
                   load_vgt::Bool = false,
                   load_pij::Bool = false,
                   load_vij::Bool = false,
                   load_sgs::Bool = false)::LagrangianDataset

    Reads a series of binary files to output a LagrangianDataset.
    The member arrays will be in format (3,3, numTsteps, numTrajs)
"""
function unpackLagrData(basepath;
                        load_vgt::Bool = false,
                        load_pij::Bool = false,
                        load_vij::Bool = false,
                        load_sgs::Bool = false)::LagrangianDataset
    
    #gives value to variable 'dataDesc'
    include(string(basepath, "datasetDescription.jl"));
    dt = dataDesc["dt"];
    filtersize = dataDesc["filtersize"];
    numTrajs = dataDesc["numTrajs"];
    numTsteps = dataDesc["numTsteps"];

    aij = zeros(1,1,1,1);
    pij = zeros(1,1,1,1);
    vij = zeros(1,1,1,1);
    sgs = zeros(1,1,1,1);

    if ((load_vgt) & (dataDesc["vgt_filename"] != nothing))
        py"""
            import numpy as np;
            aij = np.fromfile($(basepath) + $(dataDesc["vgt_filename"]), dtype=float);
            aij = aij.reshape([$(numTrajs), $(numTsteps), 3, 3]);
            """
        aij = PyArray(py"aij"o);
        aij = permutedims(aij, (3,4,2,1));
    end

    if ((load_pij) & (dataDesc["pij_filename"] != nothing))
        py"""
            import numpy as np;
            pij = np.fromfile($(basepath) + $(dataDesc["pij_filename"]), dtype=float);
            pij = pij.reshape([$(numTrajs), $(numTsteps), 3, 3]);
            """
        pij = PyArray(py"pij"o);
        pij = permutedims(pij, (3,4,2,1));
    end

    if ((load_vij) & (dataDesc["vij_filename"] != nothing))
        py"""
            import numpy as np;
            vij = np.fromfile($(basepath) + $(dataDesc["vij_filename"]), dtype=float);
            vij = vij.reshape([$(numTrajs), $(numTsteps), 3, 3]);
            """
        vij = PyArray(py"vij"o);
        vij = permutedims(vij, (3,4,2,1));
    end

    if ((load_sgs) & (dataDesc["sgs_filename"] != nothing))
        py"""
            import numpy as np;
            sgs = np.fromfile($(basepath) + $(dataDesc["sgs_filename"]), dtype=float);
            sgs = sgs.reshape([$(numTrajs), $(numTsteps), 3, 3]);
            """
        sgs = PyArray(py"sgs"o);
        sgs = permutedims(sgs, (3,4,2,1));
    end

    return LagrangianDataset(aij,
                             pij,
                             vij,
                             sgs,
                             basepath,
                             ()->unpackData(basepath,
                                            load_vgt,
                                            load_pij,
                                            load_vij,
                                            load_sgs),
                             filtersize,
                             dt);
end



function readDataDescription(filepath::String)::DataDescription
    try
        file = CSV.File(filepath, stripwhitespace=true);
        dataDesc = DataDescription(file.L,
                                   file.dt,
                                   file.Re,
                                   file.NX,
                                   file.NY,
                                   file.NZ);
        return dataDesc;
    catch
        @error "failed to read dataDescription file at location $(filepath)\n"
    end
end

# given cutout of a function f with 7 elements, centered around a point x_0 @ index 4, 
#  calculate dfdx(x_0) using 6th order, centered finite difference
function calculate_dfdx(; f::Array{Float64, 1}, dx::Float64)
    return (3/(4*dx)) * (f[5]-f[3]) -
           (3/(20*dx))* (f[6]-f[2]) +
           (1/(60*dx))* (f[7]-f[1]);
end

# given cutout of a function f with 7 elements, centered around a point x_0 @ index 4, 
#  calculate d^2f/dx^2(x_0) using 6th order, centered finite difference
function calculate_d2fdx2(; f::Array{Float64, 1}, dx::Float64)
    return (3/(2*dx^2)) * (f[5] + f[3] - 2*f[4]) -
           (3/(20*dx^2))* (f[6] + f[2] - 2*f[4]) +
           (1/(90*dx^2))* (f[7] + f[1] - 2*f[4]);
end

# given cutout of a function f with 7x7 elements, centered around a 
#  point x_0,y_0 @ index (4,4), calculate d^2f/dxdy(x_0) using 6th order,
#  centered finite difference
function calculate_d2fdxdy(; f::Array{Float64, 2}, dx::Float64, dy::Float64)
    return (2/(8*dx*dy)) * (f[5,5] + f[3,3] - f[5,3] - f[3,5]) -
           (3/(80*dx*dy))* (f[6,6] + f[2,2] - f[6,2] - f[2,6]) +
           (1/(360*dx*dy))*(f[7,7] + f[1,1] - f[7,1] - f[1,7]);
end

# calculates 6th order Lagrange poly in 1 dimension
function lagrangePoly(; i, x, xs, n=0)
    x_i = xs[n+i];

    x_js = zeros(5);
    count = 0;
    for j in -2:3
        if j != i
            count += 1;
            x_js[count] = xs[n+j];
        end
    end

    numer = prod([(x-x_js[i]) for i in 1:5]);
    denom = prod([(x_i-x_js[i]) for i in 1:5]);

    return numer/denom;
end

function interpolate_1D(;f::Array{Float64, 1},
                     xs,
                     x::Float64)
    n = findfirst(y->x<y, xs)-1;
    
    return sum([f[n-3+i]*lagrangePoly(i=-3+i, x=x, xs=xs, n=n) for i in 1:6]);
end

# given cutout of a function f with 7x7x7 elements, centered around a
#  point (x_0, y_0, z_0) @ index (4,4,4), calculate f(x) interpolated between grid points,
#  using 6th-order Lagrange polynomials
function interpolate_3D(; f::Array{Float64, 3},
                        xs,
                        ys,
                        zs,
                        x)
    n = findfirst(u->x[1]<u, xs);
    p = findfirst(u->x[2]<u, ys);
    q = findfirst(u->x[3]<u, zs);

    return sum([f[n-3+i, p-3+j, q-3+k]*
                lagrangePoly(i=-3+i, xs=xs, x=x[1], n=n)*
                lagrangePoly(i=-3+j, xs=ys, x=x[2], n=p)*
                lagrangePoly(i=-3+k, xs=zs, x=x[3], n=q)
                for i in 1:6
                for j in 1:6
                for k in 1:6]);
end

# return an array of 7 indices, centered at x_0, wrapping at boundaries
function findIndices(; x_0, L::Int=1024)
    #@assert 0 < x_0 <= L;
    inds = zeros(Int, 7);
    checkLower(i, n) = (i-n) > 0 ? (i-n) : (i-n)+L;
    checkUpper(i, n) = (i+n) <= L ? (i+n) : (i+n)-L;
    
    inds[1] = checkLower(x_0, 3);
    inds[2] = checkLower(x_0, 2);
    inds[3] = checkLower(x_0, 1);
    inds[4] = x_0;
    inds[5] = checkUpper(x_0, 1);
    inds[6] = checkUpper(x_0, 2);
    inds[7] = checkUpper(x_0, 3);

    return inds;
end

function readEularianData(basepath)::EularianDataset
    #TODO replace this by reading data description file
    dt = 3e-4;
    Re = 250;
    L = 2π;
    NX = NY = NZ = 1024;
    py"""
    import numpy as np

    u = np.fromfile('/groups/chertkov/data/eularianHITSnapshots/rstrt.0242.bin', offset=160);
    u = u.reshape([$(NZ),$(NY),$(NX)]);

    v = np.fromfile('/groups/chertkov/data/eularianHITSnapshots/rstrt.0242.2.bin');
    v = v.reshape([$(NZ),$(NY),$(NX)]);

    w = np.fromfile('/groups/chertkov/data/eularianHITSnapshots/rstrt.0242.3.bin');
    w = w.reshape([$(NZ),$(NY),$(NX)]);

    p = np.fromfile('/groups/chertkov/data/eularianHITSnapshots/rstrt.0242.4.bin')
    p = p.reshape([$(NZ),$(NY),$(NX)])
    """

    u = PyArray(py"u"o);
    v = PyArray(py"v"o);
    w = PyArray(py"w"o);
    p = PyArray(py"p"o);

    return EularianDataset(permutedims(u,[3,2,1]),
                           permutedims(v,[3,2,1]),
                           permutedims(w,[3,2,1]),
                           permutedims(p,[3,2,1]),
                           dt,
                           Re,
                           L,
                           NX,
                           NY,
                           NZ,
                           basepath);
end

function filterEularianData(ed::EularianDataset,
                            filtersize::Float64)::EularianDataset
    delta = (filtersize/ed.L)*ed.NX;
    kernel = ImageFiltering.Kernel.gaussian((delta, delta, delta));

    # pretty sure these calls are single threaded, and need ~20GB/cpu to successfully execute.
    GC.gc()
    @time filtered_u = imfilter(ed.u, kernel, Pad(:circular));
    GC.gc()
    @time filtered_v = imfilter(ed.v, kernel, Pad(:circular));
    GC.gc()
    @time filtered_w = imfilter(ed.w, kernel, Pad(:circular));
    GC.gc()
    @time filtered_p = imfilter(ed.p, kernel, Pad(:circular));
    GC.gc()

    return EularianDataset(filtered_u, filtered_v, filtered_w, filtered_p, ed.dt, ed.Re, ed.L, ed.NX, ed.NY, ed.NZ, ed.pathToSource);
end
#input an EularianDataset, output a LagrangianDataset sampled at a given dt, and spatially filtered
function sampleEularianData(ed::EularianDataset, #this can be filtered or not
                            filtersize::Float64, 
                            numSamples::Int)::LagrangianDataset

    dx = dy = dz = (ed.L/ed.NX);

    vgt = zeros(3,3,1,numSamples);
    ph = zeros(3,3,1,numSamples);

    function inRange(num)
        if (num <= 0)
            return inRange(num+ed.NX);
        elseif (num > ed.NX)
            return inRange(num-ed.NX);
        else
            return num;
        end
    end

    Threads.@threads for i in 1:numSamples
        X = [rand()*ed.L;
             rand()*ed.L;
             rand()*ed.L];

        #if NX!=NY then this is problematic
        X = inRange.(floor.(Int, (X/ed.L)*ed.NX));

        function makeNiceCutout(; f, x_0, length, dim)
            #want to return f[x_0 - e_dim*length/2 : x_0 + e_dim*length/2]
        end
        xInds = findIndices(x_0 = X[1], L=ed.NX);
        yInds = findIndices(x_0 = X[2], L=ed.NY);
        zInds = findIndices(x_0 = X[3], L=ed.NZ);
        dx = ed.L/ed.NX;
        dy = ed.L/ed.NY;
        dz = ed.L/ed.NZ;
        
        dudx = calculate_dfdx(f=ed.u[xInds, X[2], X[3]], dx=dx);
        dudy = calculate_dfdx(f=ed.u[X[1], yInds, X[3]], dx=dy);
        dudz = calculate_dfdx(f=ed.u[X[1], X[2], zInds], dx=dz);

        dvdx = calculate_dfdx(f=ed.v[xInds, X[2], X[3]], dx=dx);
        dvdy = calculate_dfdx(f=ed.v[X[1], yInds, X[3]], dx=dy);
        dvdz = calculate_dfdx(f=ed.v[X[1], X[2], zInds], dx=dz);

        dwdx = calculate_dfdx(f=ed.w[xInds, X[2], X[3]], dx=dx);
        dwdy = calculate_dfdx(f=ed.w[X[1], yInds, X[3]], dx=dy);
        dwdz = calculate_dfdx(f=ed.w[X[1], X[2], zInds], dx=dz);

        vgt[:,:,1,i] .= [dudx dudy dudz
                             dvdx dvdy dvdz
                             dwdx dwdy dwdz];

        d2Pdx2 = calculate_d2fdx2(f=ed.p[xInds, X[2], X[3]], dx=dx);
        d2Pdy2 = calculate_d2fdx2(f=ed.p[X[1], yInds, X[3]], dx=dy);
        d2Pdz2 = calculate_d2fdx2(f=ed.p[X[1], X[2], zInds], dx=dz);

        d2Pdxdy = calculate_d2fdxdy(f=ed.p[xInds, yInds, X[3]], dx=dx, dy=dy);
        d2Pdxdz = calculate_d2fdxdy(f=ed.p[xInds, X[2], zInds], dx=dx, dy=dz);
        d2Pdydz = calculate_d2fdxdy(f=ed.p[X[1], yInds, zInds], dx=dy, dy=dz);

        ph[:,:,1,i] = [d2Pdx2 d2Pdxdy d2Pdxdz
                     d2Pdxdy d2Pdy2 d2Pdydz
                     d2Pdxdz d2Pdydz d2Pdz2];
        #enforce incompressibility constraint
        diagCorrection = (1.0/3.0)*tr(ph[:,:,1,i]);
        for j in 1:3
            ph[j,j,1,i] -= diagCorrection;
        end
    end

    wnorm = sqrt(mean([norm((0.5*(vgt[:,:,1,i]-transpose(vgt[:,:,1,i])))^2) for i in 1:numSamples]));

    return LagrangianDataset(vgt,
                             ph,
                             zeros(1,1,1,1),
                             zeros(1,1,1,1),
                             ed.pathToSource,
                             ()->nothing,
                             filtersize,
                             ed.dt);
end

