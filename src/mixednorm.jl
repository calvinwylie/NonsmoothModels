using LinearAlgebra

struct MixedNorm{T <: Real}
    A :: Symmetric{T}
    B :: Symmetric{T}
    function MixedNorm(A,B)
        if size(A) != size(B)
            error("matrix sizes must match")
        # elseif !isposdef(A)
        #     lambda_min = minimum(eigvals(A))
        #     @warn string("A is not positive definite: λmin = ", lambda_min)
        # elseif !isposdef(B) 
        #     lambda_min = minimum(eigvals(B))
        #     @warn string("B is not positive definite: λmin = ", lambda_min)
        end
        return new{eltype(A)}(A,B)
    end
end

function evaluate!(model :: MixedNorm, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end

function objective(model :: MixedNorm, x)
    return sqrt(dot(x, model.A*x)) + dot(x, model.B*x)
end

function writegradient!(g, model :: MixedNorm, x)
    g[:] = (1/sqrt(dot(x, model.A*x)))*model.A*x + 2*model.B*x
    return g
end

function writehessian!(H, model :: MixedNorm, x)
    t = 1 / sqrt(dot(x, model.A*x))
    H[:,:] = t*model.A - (t^3)*model.A*x*x'*model.A' + 2*model.B
    return H
end

function writereducedhessian!(model :: MixedNorm, x, U, R)
    t = 1 / sqrt(dot(x, model.A*x))
    UAx = (U'*model.A)*x

    # println(model.A)

    E = LinearAlgebra.eigen(model.A)
    D = Diagonal((max.(E.values, 0)))
    # println(norm(model.A - (E.vectors*D*D*E.vectors'), 1 ))

    T = U'*E.vectors*(sqrt.(D))

    # Z1 = U'*model.A*(t*U)
    Z1 = t*T*T'

    # UAx = U'*E.vectors*D*E.vectors'*x

    Z2 = -(t^3)*(UAx*UAx') 
    Z3 = 2*U'*model.B*U

    # println(norm(Z1 - Z1', 1))
    # R[:,:] =  t*(U'*model.A*U) - (t^3)*(UAx*UAx') + 2*U'*model.B*U
    R[:,:] =  Z1 + Z2 + Z3
    # R[:,:] = (1/2)*(R' + R)
    return R
end
