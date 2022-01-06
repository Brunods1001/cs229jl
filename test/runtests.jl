using Test
using cs229

@testset "dot product" begin
    a = [1, 2, 3]
    b = [10, 11, 12]
    ab = cs229.dot(a, b)
    @test ab == (1 * 10 + 2 * 11 + 3 * 12)
    A = [1  2; 3  4; 5  6]
    B = [1; 2]
    AB = cs229.dot(A, B)
    @test AB == [5; 11; 17]
end

@testset "feature scaling" begin
    x1 = [89, 72, 94, 69]
    x2 = x1.^2
    y = [96, 74, 87, 78]
    x0 = ones(length(x1))
    X = [x1 x2]
    X_scaled = cs229.mean_norm(cs229.range_scaling(X))
    X_scaled = [X_scaled x0]
    @test X_scaled[3, 1] == 0.52
end

@testset "ex1" begin
    ex1data1 = strip(read("../hw/ex1/ex1data1.txt", String))
    lines = [tryparse.(Float64, line) for line in split.(split(ex1data1, "\n"), ",")];
    dat = hcat(lines...)';
    x = dat[:, 1]
    X = [ones(length(x)) x]
    y = dat[:, 2]
    @test size(X) == (97, 2)
    @test size(y) == (97,)
    ans = 32.07
    ans2 = 54.24
    iterations = 1500
    alpha = 0.01
    θ = zeros(2, 1)
    @testset "compute cost" begin
        cost = cs229.compute_cost(X, y, θ)
        @test round(cost; digits=2) == ans
        cost2 = cs229.compute_cost(X, y, [-1; 2])
        @test round(cost2; digits=2) == ans2
    end
    @testset "gradient descent" begin
        ans = [-3.6303; 1.1664];
        new_theta = cs229.gradient_descent(X, y, θ, alpha, iterations)
        @test all(round.(new_theta; digits=4) .== ans)
    end
end

@testset "ex1 part2" begin
    ex1data2 = strip(read("../hw/ex1/ex1data2.txt", String))
    lines = [tryparse.(Float64, line) for line in split.(split(ex1data2, "\n"), ",")];
    dat = hcat(lines...)';
    x = dat[:, 1:2]
    X_norm, mu, sigma  = cs229.standardize(x)
    X = [ones(size(x)[1]) X_norm]
    y = dat[:, 3]

    @test size(X) == (47, 3)
    @test size(y) == (47,)

    @testset "feature normalization" begin
        ans_mu = [2.0007e+03, 3.1702e+00]
        ans_sigma = [794.7024, 0.7610]
        @test isapprox(ans_mu, mu; rtol=0.001)
        @test isapprox(ans_sigma, sigma; rtol=0.001)
        @test isapprox(X_norm[1, :], [0.1300, -0.2237]; rtol=0.001)
    end

    @testset "gradient descent" begin
        ans = [3.3430e+05; 1.0009e+05; 3.6735e+03]
        iterations = 400
        alpha = 0.01
        θ = zeros(3, 1)
        new_θ = cs229.gradient_descent(X, y, θ, alpha, iterations)
        @test isapprox(new_θ, ans; rtol=0.001)
    end

    @testset "normal equations" begin
        ans = [8.9598e+04; 1.3921e+02; -8.7380e+03]
        X = [ones(size(x)[1]) x]
        new_θ = cs229.normal_equation(X, y)
        @test isapprox(new_θ, ans; rtol=0.001)
    end
end
