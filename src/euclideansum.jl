struct EuclideanSum
    structuremodels :: Array{Any,1}
end

function evaluate!(model :: EuclideanSum, x, g=missing, H=missing)
    V = structure_values(model, x)
    S = sign.(V)
    if !ismissing(g)
        writegradient!(g, model, S, x)
    end
    if !ismissing(H)
        writehessian!(H, model, S, x)
    end
    return sum(abs.(V))
end

function objective(model :: EuclideanSum, x)
    return sum(abs.(structure_values(model, x)))
end

function writegradient!(g, model :: EuclideanSum, signs, x)
    g_tmp = similar(g)
    for i = 1:length(model.structuremodels)
        writegradient!(g_tmp, model.structuremodels[i], x)
        g .+= signs[i]*g_tmp
    end
    return g
end

function writehessian!(H, model :: EuclideanSum, signs, x)
    H_tmp = similar(H)
    for i = 1:length(model.structuremodels)
        writehessian!(H_tmp, model.structuremodels[i], x)
        H .+= signs[i]*H_tmp
    end
    return H
end

function structure_values(model :: EuclideanSum, x)
    return [objective(m, x) for m in model.structuremodels]
end