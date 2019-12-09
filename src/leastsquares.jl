using LinearAlgebra

struct LeastSquares
    A
    b
    AtA
    Atb 
    LeastSquares(A, b) = new(A, b, A'*A, A'*b)
end

function evaluate!(model :: LeastSquares, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end


function objective(model :: LeastSquares, x)
    return (1/2)*norm(model.A*x - model.b)^2
end

function writegradient!(g, model :: LeastSquares, x)
    g[:] = model.AtA*x - model.Atb
    return g
end

function writehessian!(H, model :: LeastSquares, x)
    H[:,:] = model.AtA
    return H
end