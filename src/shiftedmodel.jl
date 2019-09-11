struct ShiftedModel
    internalmodel
    shift
end

function evaluate!(model :: ShiftedModel, x, g=missing, H=missing)
    if !ismissing(g)
        writegradient!(g, model.internalmodel, x - model.shift)
    end
    if !ismissing(H)
        writehessian!(H, model.internalmodel, x - model.shift)
    end
    return objective(model.internalmodel, x - model.shift)
end

function objective(model :: ShiftedModel, x)
    return objective(model.internalmodel, x - model.shift)
end

# function gradient(model :: ShiftedModel, x)
#     return gradient(model.internalmodel, x - model.shift)
# end

function writegradient!(g, model :: ShiftedModel, x)
    return writegradient!(g, model.internalmodel, x - model.shift)
end

# function hessian(model :: ShiftedModel, x)
#     return hessian(model.internalmodel, x - model.shift)
# end

function writehessian!(H, model :: ShiftedModel, x)
    return writehessian!(H, model.internalmodel, x - model.shift)
end

function writereducedhessian!(model :: ShiftedModel, x, U, R)
    return writereducedhessian!(model.internalmodel, x - model.shift, U, R)
end
