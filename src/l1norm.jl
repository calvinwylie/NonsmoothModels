using LinearAlgebra

struct L1Norm{T <: Real}
    lambda :: T
end

function objective(model :: L1Norm, x)
    return model.lambda*norm(x, 1)
end

function prox!(p, model :: L1Norm, x, t)
    p[:] = max.(abs.(x) .- t*model.lambda, 0) .* sign.(x)
    return p
end

# Solves $v \in A*w + D(\partial |x|_1)*w$ for w
# where D is the graphical derivative 
function subgradientderivativesolve!(w, model :: L1Norm, x, A, v)
    w .= 0
    # C = cholesky(A[x .!= 0, x .!= 0])
    # w[x .!= 0] = C \ v[x .!= 0]
    w[x .!= 0] = A[x .!= 0, x .!= 0] \ v[x .!= 0]
end