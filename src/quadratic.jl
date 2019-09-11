using LinearAlgebra

struct Quadratic{T <: Real}
    a :: T
    g :: Vector{T}
    H :: Symmetric{T, Matrix{T}}
end

function evaluate!(model :: Quadratic, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end


function objective(model :: Quadratic, x)
    return model.a + dot(model.g,x) + (1/2)*dot(x, model.H*x)
end

function writegradient!(g, model :: Quadratic, x)
    g[:] = model.g + model.H*x
    return g
end

function writehessian!(H, model :: Quadratic, x)
    H[:,:] = model.H
    return H
end