using LinearAlgebra

struct LogisticLoss
    D
    y
    LogisticLoss(data, labels) = new(data, labels)
end

sigma(x) = exp(x)/(1 + exp(x))

function evaluate!(model :: LogisticLoss, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model, x)
    end
    if !ismissing(H)
        writehessian!(H, model, x)
    end
    return objective(model, x)
end


function objective(model :: LogisticLoss, x)
    N = size(model.D, 1)
    return (1/N)*sum(log(1 + exp(-model.y[i]*dot(model.D[i,:], x))) for i = 1:N)
end

function writegradient!(g, model :: LogisticLoss, x)
    N = size(model.D, 1)
    g[:] = -(1/N)*model.D'*[sigma(-model.y[i]*dot(model.D[i,:], x))*model.y[i] for i = 1:N]
    return g
end

function writehessian!(H, model :: LogisticLoss, x)
    N = size(model.D, 1)
    e = [sigma(-model.y[i]*dot(model.D[i,:], x))*(1 - sigma(-model.y[i]*dot(model.D[i,:], x))) for i = 1:N]
    H[:,:] = (1/N)*model.D'*Diagonal(e)*model.D
    return H
end