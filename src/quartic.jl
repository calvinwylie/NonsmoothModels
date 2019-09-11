using LinearAlgebra

struct Quartic{T <: Real}
    a :: T
    g :: Vector{T}
    H :: Symmetric{T, Matrix{T}}
    c :: T
end

function evaluate!(model :: Quartic, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end


function objective(model :: Quartic, x)
    return model.a + dot(model.g,x) + (1/2)*dot(x, model.H*x) + (model.c/4)*dot(x, x)^2
end

function writegradient!(g, model :: Quartic, x)
    g[:] = model.g + model.H*x + model.c*dot(x, x)*x
    return g
end

function writehessian!(H, model :: Quartic, x)
    H[:,:] = model.H + model.c*(2*x*x' + dot(x, x)*I)
    return H
end