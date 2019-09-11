struct NesterovCR
    n::Integer
    NesterovCR(n) = n < 2 ? error("n must be > 1") : new(n)
end

function evaluate!(model :: NesterovCR, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end

function objective(model :: NesterovCR, x)
    n = length(x)
    return (1/4)*(x[1]-1)^2 + sum(abs(x[i+1] - 2*x[i]^2 + 1) for i = 1:n-1)
end

function writegradient!(g, model :: NesterovCR, x)
    n = model.n
    g .= 0
    g[1] = (1/2)*(x[1] - 1)
    for i = 1:n-1
        s = sign(x[i+1] - 2*x[i]^2 + 1)
        g[i+1] += s*1
        g[i] += s*(-4*x[i])
    end

    return g
end

function writehessian!(H, model :: NesterovCR, x)
    n = model.n
    H .= 0

    H[1,1] = 1/2

    for i = 1:n-1
        s = sign(x[i+1] - 2*x[i]^2 + 1)
        H[i+1,i] += s
        H[i,i+1] += s
        H[i,i] += -4*s
    end
    return H
end