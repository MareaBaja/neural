using DataFrames
using MLDatasets 
using LinearAlgebra
using Printf
using Random
using Statistics

# ----- DATA --------

dataset_train = MNIST(:train)
dataset_test

# ----- Network variables --------

c1 = zeros(Float32, 784) 
weights1 = rand(Float32, 16, 784)
biases1 = rand(Float32, 16)
c2 = zeros(Float32, 16)
weights2 = rand(Float32, 16, 16)
biases2 = rand(Float32, 16)
c3 = zeros(Float32, 16)
weights3 = rand(Float32, 10, 16)
biases3 = rand(Float32, 10)
c4 = zeros(Float32, 10)

NETWORK = Dict(1 => [c1, weights1, biases1], 2 => [c2, weights2, biases2],
    3 => [c3, weights3, biases3], 4 => [c4])

# ----- FUNCTIONS --------

function compute_layer(weights, v, b)
    Y = zeros(Float32, length(b))
    Y = mul!(Y, weights, v) + b
    broadcast(relu, Y) #aplica relu a cada elemento del vector, no hace falta el return
end

function relu(x)
    return max(0, x)
end

# Mean squared error: takes vector (output layer) and computes
# mean squared error with respect to target value
square(x) = x^2
function cost(final_layer, target)::Float32 #Esto asegura que el tipo que devuelve sea un Float32
    target_vector = zeros(Float32, length(final_layer))
    target_vector[target+1] = 1
    sum(broadcast(square, final_layer .- target_vector))
end

relu(x::Number) = max(0, x)
function relu(x::Vector)
    broadcast(relu, x)
end

function f_propagation(image::Matrix, net::Dict)
    """Perform forward propagation by defining input layer as
    determined by input x and propagating through predefined
    empty network."""
    # Set input layer
    net[1][1] = vec(image)
    for i in 2:length(net)
        net[i][1] = compute_layer(
            net[i-1][2],
            net[i-1][1],
            net[i-1][3])
    end
    return net
end

print(typeof(axes(dataset_train.features)[3]))
print(length(axes(dataset_train.features)[3]))

print(NETWORK[4][1])

function training(dataset_training::MNIST, net::Dict) #images tiene las imagenes y un indice (de 1 a 60000)
    #MNIST es el tipo de dataset_training
    costs = zeros(Float32, 60000)
    @time for i in 1:length(dataset_training.targets) #con axes se accede a la tercera dimension de images, que es el indice
        net = f_propagation(dataset_training[i].features, net)
        costs[i] = cost(net[length(net)][1], dataset_training[i].targets)
        print(string(i) * "\n")
    end
    return mean(costs)
end

# https://docs.julialang.org/en/v1/manual/parallel-computing/
# https://julialang.org/blog/2019/07/multithreading/

print(dataset_train[1].targets)
training(dataset_train.features)

network = f_propagation(dataset_train[1].features, NETWORK)
cos = cost(network[length(network)][1], dataset_train[1][2])

average_cost = training(dataset_train, NETWORK)
#funcion que calcule el costo
#funcion que calcule el gradiente