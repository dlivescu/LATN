
@testset "1D Lagrange Polynomial Interpolation" begin
    dx = rand();
    xs = rand():dx:(rand()+1)*10;
    f = atan.(xs);
    tol = (((7*dx)^7)/(factorial(7)))*(π/2);
    for i in 1:10000
        x = rand()*(xs[end-4]-xs[1+4])+xs[1+4];
        @test abs(interpolate_1D(f=f,xs=xs,x=x)-atan(x)) <= tol;
    end
end


@testset "1D Lagrange Polynomial Interpolation" begin
    dx = rand();
    xs = rand():dx:(rand()+1)*10;
    f = atan.(xs);
    tol = (((7*dx)^7)/(factorial(7)))*(π/2);
    for i in 1:10000
        x = rand()*(xs[end-4]-xs[1+4])+xs[1+4];
        @test abs(interpolate_1D(f=f,xs=xs,x=x)-atan(x)) <= tol;
    end
end

@testset "3D Lagrange Polynomial Interpolation" begin
    dx, dy, dz = [rand(), rand(), rand()];
    xs = rand():dx:(rand()+1)*10;
    ys = rand():dy:(rand()+1)*10;
    zs = rand():dz:(rand()+1)*10;

    tol = (((7*min(dx,dy,dz))^7)/(factorial(7)))*(π/2)*maximum(ys)^2*1.0;

    f = zeros(length(xs), length(ys), length(zs));
    for i in 1:length(xs)
        for j in 1:length(ys)
            for k in 1:length(zs)
                f[i,j,k] = atan(xs[i])*(ys[j]^2)*besselj(0,zs[k]);
            end
        end
    end

    for i in 1:1000
        x = [rand()*(xs[end-4]-xs[1+4])+xs[1+4];
             rand()*(ys[end-4]-ys[1+4])+ys[1+4];
             rand()*(zs[end-4]-zs[1+4])+zs[1+4]];
        truth = atan(x[1])*(x[2]^2)*besselj(0,x[3]);
        pred = interpolate_3D(f=f,xs=xs,ys=ys,zs=zs,x=x);
        @test abs(truth-pred) <= tol;
    end        
end
