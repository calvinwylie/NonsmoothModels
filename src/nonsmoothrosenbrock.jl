struct NonsmoothRosenbrock
    w :: Real
    NonsmoothRosenbrock(w) = w >= 0 ? new(w) : error("w must be >= 0")
end

function evaluate!(model :: NonsmoothRosenbrock, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end

function objective(model :: NonsmoothRosenbrock, x)
    n = length(x)
    return model.w*(x[1] - 1)^2 + sum(abs.(x[2:n] .- x[1:n-1].^2))
end

function writegradient!(g, model :: NonsmoothRosenbrock, x)
    n = length(x)
    S = sign.(x[2:n] .- x[1:n-1].^2)
    g .= 0
    g[1] = 2*model.w*(x[1] - 1)
    for i = 1:n-1
        g[i+1] += S[i]
        g[i] += -2*S[i]*x[i]
    end
    return g
end

function writehessian!(H, model :: NonsmoothRosenbrock, x)
    n = length(x)
    S = sign.(x[2:n] .- x[1:n-1].^2)
    H .= 0
    H[1,1] = 2*model.w
    for i = 1:n-1
        H[i,i] += -2*S[i]
    end
    return H
end
