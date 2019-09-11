using LinearAlgebra

struct MaxEig{T <: Real}
    A0 :: Symmetric{T, Matrix{T}}
    # A :: Array{Symmetric{T, Matrix{T}},1}
    A :: Array{T,3}
    H :: Matrix{T}
    MaxEig(A0, A) = new{eltype(A0)}(A0, cat(A..., dims=3), zeros(eltype(A0), length(A), length(A)))
end

Base.show(io :: IO, model :: MaxEig) = print(io, "(", size(model.A0, 1), " x ", size(model.A0, 2), ") MaxEig with ", size(model.A, 3), " variables")

# function evaluate(model :: MaxEig{Float64}, x :: Vector{Float64}; computegradient=true, computehessian=true)
#     E = eigen(model, x)
#     return (objective = objective(model, E), 
#             gradient = computegradient ? gradient(model, E) : missing, 
#             hessian = computehessian ? hessian(model, E) : missing)
# end

function evaluate!(model :: MaxEig{Float64}, x :: Vector{Float64}, g=missing, H=missing)
    E = eigen(model, x)
    if !ismissing(g)
        writegradient!(g, model, E)
    end
    if !ismissing(H)
        writehessian!(H, model, E)
    end
    return objective(model, E)
end

function objective(model :: MaxEig{Float64}, x :: Vector{Float64})
    return objective(model, eigen(model, x))
end

# function gradient(model :: MaxEig{Float64}, x :: Vector{Float64})
#     n = length(x)
#     g = zeros(Float64, n)
#     return writegradient!(model, eigen(model, x), g)
# end

function writegradient!(g, model :: MaxEig{Float64}, x :: Vector{Float64})
    return writegradient!(g, model, eigen(model, x))
end

# function hessian(model :: MaxEig{Float64}, x :: Vector{Float64})
#     n = length(x)
#     H = zeros(Float64, n, n)
#     return writehessian!(model, eigen(model, x), H)
# end

function writehessian!(H, model :: MaxEig{Float64}, x :: Vector{Float64})
    return writehessian!(H, model, eigen(model, x))
end

function writereducedhessian!(model :: MaxEig{Float64}, x :: Vector{Float64}, U, R)
    return writereducedhessian!(model, eigen(model, x), U, R)
end

function isdifferentiable(model :: MaxEig{Float64}, x)
    U,_ = eigen(model.A0 + linearcombination(x, model.A))
    sort!(U, true)
    !isapprox(U[1], U[2]; atol=eps(typeof(U[1])), rtol=0)
end

function eigen(model :: MaxEig{Float64}, x :: Vector{Float64})
    B = linearcombination(x, model.A) :: Matrix{Float64}
    return LinearAlgebra.eigen(model.A0 + B)
end

function linearcombination(x, A)
    @views return sum(x[i]*A[:,:,i] for i in 1:length(x))
end

function objective(model :: MaxEig, E :: Eigen)
    return maximum(E.values)
end

# function gradient(model :: MaxEig, E :: Eigen)
#     n = size(model.A, 3)
#     @views y = E.vectors[:, argmax(E.values)]
#     @views g = [dot(y, model.A[:,:,i]*y) for i=1:n]
#     return g
# end

function writegradient!(g, model :: MaxEig, E :: Eigen)
    n = size(model.A, 3)
    m = length(E.values)

    y = view(E.vectors, :, m)
    for i = 1:n
        g[i] = dot(y, view(model.A, :, :, i)*y)
    end
    return g
end

# function hessian(model :: MaxEig, E :: Eigen)
#     n = size(model.A, 3)
#     H = zeros(eltype(E.values), n, n)
    # lambda = E.values
    # Q = E.vectors
    # i = argmax(lambda)
    # tmp = similar(lambda)
    # @inbounds @views for k = 1:n, j = k:n
    #     local entry::Float64 = 0
    #     for s = 1:length(lambda)
    #         if s != i
    #             # entry += dot(Q[:,i], model.A[:,:,k]*Q[:,s])*dot(Q[:,i], model.A[:,:,j]*Q[:,s]) / (lambda[i] - lambda[s])
    #             e1 = dot(Q[:,i], mul!(tmp, model.A[:,:,k], Q[:,s]))
    #             e2 = dot(Q[:,i], mul!(tmp, model.A[:,:,j], Q[:,s]))
    #             entry += 2*e1*e2 / (lambda[i] - lambda[s])
    #         end
    #     end
    #     # entry =  2*sum((Q[:,i]'*model.A[k]*Q[:,s]*Q[:,i]'*model.A[j]*Q[:,s]) / (lambda[i] - lambda[s]) for s in eachindex(lambda) if s != i)
    #     H[k,j] = entry
    #     H[j,k] = entry
    # end
    # return H
# end

function writehessian!(H, model :: MaxEig, E :: Eigen)
    n = size(model.A, 3)
    m = length(E.values)

    tmp = zeros(eltype(model.A0), m-1, n)
    @views for k = 1:n
        tmp[:, k] = (model.A[:,:,k]*E.vectors[:,1:m-1])'*E.vectors[:,m]
    end
    H[:,:] = 2*tmp'*(tmp ./ (E.values[m] .- view(E.values, 1:m-1)))
    return H
end

function writereducedhessian!(model :: MaxEig, E :: Eigen, U, R)
    n = size(model.A, 3)
    m = length(E.values)

    tmp = zeros(eltype(model.A0), m-1, n)
    @views for k = 1:n
        tmp[:, k] = (model.A[:,:,k]*E.vectors[:,1:m-1])'*E.vectors[:,m]
    end
    T = tmp*U
    R[:,:] = 2*T'*(T ./ (E.values[m] .- view(E.values, 1:m-1)))
    # R[:,:] = (1/2)*(R + R')
    return R
end

# function reducedhessianvecprod(model :: MaxEig, x, y, U)
#     E = eigen(model, x)

#     n = size(model.A, 3)
#     m = length(E.values)

#     tmp = zeros(eltype(model.A0), m-1, n)
#     @views for k = 1:n
#         tmp[:, k] = (model.A[:,:,k]*E.vectors[:,1:m-1])'*E.vectors[:,m]
#     end
#     return 2*(U'*tmp')*((tmp*y) ./ (E.values[m] .- view(E.values, 1:m-1)))

# end
