# NonsmoothModels

Nonsmooth functions and optimization models implemented in Julia for research purposes.

## Basic models

### Quadratic functions
![equation](https://latex.codecogs.com/svg.latex?a%20&plus;%20g%5ETx%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%20x%5ET%20H%20x)

`model = Quadratic(a::T, g::Vector{T}, H::Symmetric{T})`

### Quartic functions
![equation](https://latex.codecogs.com/svg.latex?a%20&plus;%20g%5ETx%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%20x%5ET%20H%20x%20&plus;%20%5Cfrac%7Bc%7D%7B4%7D%7C%7Cx%7C%7C%5E4)

`model = Quartic(a::T, g::Vector{T}, H::Symmetric{T}, c::T)`

### L1 Norm
![equation](https://latex.codecogs.com/svg.latex?%5Clambda%20%5C%7Cx%5C%7C_1)

`model = L1Norm(lambda::Real)`

### Mixed norm functions
![equation](https://latex.codecogs.com/svg.latex?%5Csqrt%7Bx%5ETAx%7D%20&plus;%20x%5ETBx)

`model = MixedNorm(A::Symmetric{T}, B::Symmetric{T})`

### Maximum eigenvalue functions
![equation](https://latex.codecogs.com/svg.latex?%5Clambda_%7B%5Cmax%7D%5CBig%28A_0%20&plus;%20%5Csum_%7Bi%3D1%7D%5Emx_iA_i%5CBig%29)

`model = MaxEig(A0::Symmetric{T}, A::Array{T,3})`

### Least squares loss
![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B1%7D%7B2%7D%20%5C%7CAx%20-%20b%5C%7C%5E2)

`model = LeastSquares(A, b)`

### Logistic loss
![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B1%7D%7BN%7D%20%5Csum_%7Bi%3D1%7D%5EN%20%5Clog%281%20&plus;%20e%5E%7B-y_i%20d_i%5ETx%7D%29)

`model = LogisticLoss(D, y)`

### Nonsmooth Rosenbrock function
![equation](https://latex.codecogs.com/svg.latex?w%28x_1%20-%201%29%5E2%20&plus;%20%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D%5Cbig%7Cx_%7Bi&plus;1%7D%20-%20x_i%5E2%5Cbig%7C)

`model = NonsmoothRosenbrock(w)`

### Nesterov nonsmooth Chebyshev–Rosenbrock function
![equation](https://latex.codecogs.com/svg.latex?%5Cfrac%7B1%7D%7B4%7D%28x_1%20-%201%29%5E2%20&plus;%20%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D%5Cbig%7Cx_%7Bi&plus;1%7D%20-%202x_i%5E2%20&plus;%201%20%5Cbig%7C)

`model = NesterovCR(n)`

### Chained CB3 II
![equation](https://latex.codecogs.com/svg.latex?%5Cmax%20%5CBig%28%20%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D%20x_i%5E4%20-%20x_%7Bi&plus;1%7D%5E2%2C%20%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D%20%282-x_i%29%5E2%20&plus;%20%282-x_%7Bi&plus;1%7D%29%5E2%2C%20%5Csum_%7Bi%3D1%7D%5E%7Bn-1%7D%202%20e%5E%7Bx_%7Bi&plus;1%7D%20-%20x_i%7D%20%5CBig%29)

`model = ChainedCB3II(n)`

## Composite models

### Pointwise maximum
![equation](https://latex.codecogs.com/svg.latex?%5Cmax_%7B1%20%5Cleq%20i%20%5Cleq%20k%7D%20f_i%28x%29)

`model = MaxFunction([f for f in list_of_models])`

### Euclidean sum
![equation](https://latex.codecogs.com/svg.latex?%5Csum_%7Bi%3D1%7D%5Ek%20%7Cf_i%28x%29%7C)

`model = EuclideanSum([f for f in list_of_models])`

### Shifted model
![equation](https://latex.codecogs.com/svg.latex?f%28x%20-%20z%29)

`model = ShiftedModel(original_model, z)`
