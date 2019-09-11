struct Ferrier
end

function evaluate!(model :: Ferrier, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end

function objective(model :: Ferrier, x)
    n = length(x)
    return sum(abs.(((1:n).*(x.^2) .- 2*x) .+ sum(x))) + (1/2)*norm(x,2)^2
end

function writegradient!(g, model :: Ferrier, x)
    n = length(x)
    S = sign.((1:n).*(x.^2) .- 2*x .+ sum(x))
    for i = 1:n
        g[i] = S[i]*(2*i*x[i] - 2) + sum(S) + x[i]
    end
    return g
end

function writehessian!(H, model :: Ferrier, x)
    n = length(x)
    S = sign.((1:n).*(x.^2) .- 2*x .+ sum(x))
    for i = 1:n
        H[i,i] = S[i]*2*i + 1
    end
    return H
end

