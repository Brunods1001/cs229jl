module cs229
import LinearAlgebra

function dot(a, b)
    if (length(size(a)) > 1) | (length(size(b)) > 1)
        return a * b
    else
        return LinearAlgebra.dot(a, b)
    end
end

function feature_scaling(X, scaling_function)
    scaled_cols = scaling_function.(cols)
end

range_x(x) = maximum(x) - minimum(x)
mean(x) = sum(x) / length(x)
function std(x; ispop=false)
    if ispop
        sqrt(1 / length(x) * sum((x .- mean(x)).^2))
    else
        sqrt(1 / (length(x) - 1) * sum((x .- mean(x)).^2))
    end
end
scale_max_x(x) = x ./ maximum(x)

"takes in a matrix"
function mean_norm(X)
    cols = eachcol(X)
    scales = mean.(cols)
    scaled_cols = [col .- scales[i] for (i, col) in enumerate(cols)]
    hcat(scaled_cols...)
end

function standardize(X)
    cols = eachcol(X)
    scales_mean = mean.(cols)
    scales_std = std.(cols, ispop=false)
    scaled_cols = [(col .- scales_mean[i]) / scales_std[i] for (i, col) in enumerate(cols)]
    hcat(scaled_cols...), scales_mean, scales_std
end


function range_scaling(X)
    cols = eachcol(X)
    scales = range_x.(cols)
    scaled_cols = cols ./ scales
    hcat(scaled_cols...)
end


function max_norm(X)
    cols = eachcol(X)
    scales = maximum.(cols)
    scaled_cols = cols ./ scales
    hcat(scaled_cols...)
end


function mean_max_norm(X)
    cols = eachcol(X)
    scales_max = maximum.(cols)
    scales_mean = mean.(cols)
    scaled_cols_mean = [col .- scales_mean[i] for (i, col) in enumerate(cols)]
    scaled_cols = scaled_cols_mean ./ scales_max
    hcat(scaled_cols...)
end

function least_squares(y, ŷ)
    m = length(y)
    first(1 / 2m * (y - ŷ)' * (y - ŷ))
end

function regress(X, θ)
    X * θ
end

function compute_cost(
        X, y, θ; 
        hypothesis=regress, 
        f_cost=least_squares)
    ŷ = hypothesis(X, θ)
    f_cost(y, ŷ)
end

function gradient_descent(X, y, theta, alpha, num_iters)
    m = length(y)
    for _ in 1:num_iters
        y_pred = X * theta;
        theta = theta - 1 / m * alpha * X' * (y_pred - y);
    end
    reshape(theta, :, 1)
end

function normal_equation(X, y)
    inv(X' * X) * X' * y
end

end # module

