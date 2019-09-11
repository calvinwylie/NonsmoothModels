struct MaxFunction
    structuremodels :: Array{Any,1}
end

function evaluate!(model :: MaxFunction, x, g=missing, H=missing)
    t = argmax(structure_values(model, x))
    if !ismissing(g)
        writegradient!(g, model.structuremodels[t], x)
    end
    if !ismissing(H)
        writehessian!(H, model.structuremodels[t], x)
    end
    return objective(model.structuremodels[t], x)
end

function objective(model :: MaxFunction, x)
    t = argmax(structure_values(model, x))
    return objective(model.structuremodels[t], x)
end

function structure_values(model :: MaxFunction, x)
    return [objective(m, x) for m in model.structuremodels]
end
