module NonsmoothModels

export NesterovCR
include("nesterovcr.jl")

export MixedNorm
include("mixednorm.jl")

export NonsmoothRosenbrock
include("nonsmoothrosenbrock.jl")

export ChainedCB3II
include("chainedcb3ii.jl")

export Ferrier
include("ferrier.jl")

export MaxEig
include("maxeig.jl")

export ShiftedModel
include("shiftedmodel.jl")

export Quadratic
include("quadratic.jl")

export Quartic
include("quartic.jl")

export MaxFunction
include("maxfunction.jl")

export EuclideanSum
include("euclideansum.jl")

export LeastSquares
include("leastsquares.jl")

export LogisticLoss
include("logistic.jl")

export L1Norm
include("l1norm.jl")

export evaluate!, objective
export writegradient!, writehessian!, writereducedhessian!
export prox!
export subgradientderivativesolve!

end