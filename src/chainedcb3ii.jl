struct ChainedCB3II
    n::Integer
    ChainedCB3II(n) = n < 2 ? error("n must be > 1") : new(n)
end

function evaluate!(model :: ChainedCB3II, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end

function structure_values(model :: ChainedCB3II, x)
    n = model.n
    c1 = sum(x[1:n-1].^4 + x[2:n].^2)
    c2 = sum((2 .- x[1:n-1]).^2 + (2 .- x[2:n]).^2)
    c3 = sum(2*exp.(x[2:n] .- x[1:n-1]))

    return c1, c2, c3
end

function objective(model :: ChainedCB3II, x)
    return maximum(structure_values(model, x))
end

function writegradient!(g, model :: ChainedCB3II, x)
    n = model.n
    t = argmax(structure_values(model, x))
    if t == 1
        g[1] = 4*x[1]^3
        g[2:n-1] .= 4*x[2:n-1].^3 + 2*x[2:n-1]
        g[n] = 2*x[n]
    elseif t == 2
        g[1] = 2*x[1] - 4
        g[2:n-1] .= 4*x[2:n-1] .- 8
        g[n] = 2*x[n] - 4
    else
        g[1] = -2*exp(x[2] - x[1])
        g[2:n-1] .= 2*exp.(x[2:n-1] .- x[1:n-2]) - 2*exp.(x[3:n] .- x[2:n-1])
        g[n] = 2*exp(x[n] - x[n-1])
    end

    return g
end

function writehessian!(H, model :: ChainedCB3II, x)
    n = model.n
    t = argmax(structure_values(model, x))
    if t == 1
        H[1,1] = 12*x[1]^2
        for j = 2:n-1
            H[j,j] = 12*x[j]^2 + 2
        end
        H[n,n] = 2
    elseif t == 2
        H[1,1] = 2
        for j = 2:n-1
            H[j,j] = 4
        end
        H[n,n] = 2
    else
        H[1,1] = 2*exp(x[2] - x[1])
        for i = 2:n-1
            e1  = -2*exp(x[i] - x[i-1])
            H[i, i-1] = e1
            H[i-1, i] = e1

            e2 = 2*exp(x[i+1] - x[i]) + 2*exp(x[i] - x[i-1])
            H[i,i] = e2

            e3 = -2*exp(x[i+1] - x[i])
            H[i, i+1] = e3
            H[i+1, i] = e3
        end
        H[n,n] = 2*exp(x[n] - x[n-1])
    end

    return H
end
