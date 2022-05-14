#Sihao Ren // sren03 // 947105531
using Evolutionary
using Flux
using Flux: onehot, onecold, logitcrossentropy, onehotbatch, crossentropy 
using MLDatasets
using Random
using Statistics
using MLJBase
using Printf
using BSON: @load # for load weights
using Plots
using DelimitedFiles
import Evolutionary.initial_population
using Zygote
import Evolutionary.NonDifferentiable
import Base: copy, copyto!

#random seed for Evolutionary
Random.seed!(63456345)

#Read in and balance data
rawdata = readdlm("data/test2.csv",',',skipstart=1)';
one = 0
zero = 0
for i in rawdata[20,:]
    if i == 1
        one = one + 1
    else
        zero = zero + 1
    end
end

difference = one - zero

function newrand()
    randCol = zeros(0)
    push!(randCol, rand(10.0:99.0))
    push!(randCol, rand(5.3:40.2))
    push!(randCol, rand(1.1:15.1))
    push!(randCol, rand(0.1:8.1))
    push!(randCol, rand(1.1:15.1))
    push!(randCol, rand(30.1:60.1))
    push!(randCol, rand(0.1:0.9))
    push!(randCol, rand(0.1:4.1))
    push!(randCol, rand(20.1:50.1))
    push!(randCol, rand(0.1:5.1))
    push!(randCol, rand(0.1:7.1))
    push!(randCol, rand(45.1:99.1))
    push!(randCol, rand(0.1:5.1))
    push!(randCol, rand(0.1:5.1))
    push!(randCol, rand(0.1:7.1))
    push!(randCol, rand(0.1:7.1))
    push!(randCol, rand(0.1:3.1))
    push!(randCol, rand(0.1:3.1))
    push!(randCol, rand(0.1:5.1))
    if rand(0:1) == 0
        push!(randCol, 0.0)
    else
        push!(randCol, 1.0)
    end
    return randCol
end

for i = 1:size(rawdata,2)
    if rawdata[20,i] == 0 && difference > 0
        rawdata = hcat(rawdata, rawdata[:,i])
        rawdata = hcat(rawdata, newrand())

        difference = difference - 1
    end
end
filldata = rawdata[ :, shuffle(1:end)];

x = filldata[1:19, :]
y = filldata[20, :];

x_train = x[:,1:floor(Int, size(x,2)*0.7)]
y_train = y[1:floor(Int, size(x,2)*0.7)]
x_test = x[:,floor(Int, size(x,2)*0.7)+1:end]
y_test = y[floor(Int, size(x,2)*0.7)+1:end];

#zip feature & label together using zip instead of batch for Evolutionary function
train_data = [ (x, onehot(l, unique(y_train))) for (x, l) in zip(eachcol(x_train), y_train)]
test_data = [ (x, onehot(l, unique(y_test))) for (x, l) in zip(eachcol(x_test), y_test)];

accuracy(model,x,y) = mean(onecold(model(x)) .== onecold(y))
accuracy(xy, model) = mean( onecold(model(x)) .== onecold(y) for (x,y) in xy)

loss(model) = (x,y)->logitcrossentropy(model(x), y)
loss(model,x,y) = loss(model)(x, y)
loss(xy, model) = loss(model)(hcat(map(first,xy)...), hcat(map(last,xy)...))  

fitness(m) = loss(train_data, m)
#flat the weight and bias for evolutionary function to evolutionary
function uniform_mlp(m1::T, m2::T) where {T <: Chain}
    θ1, re1 = Flux.destructure(m1);
    θ2, re2 = Flux.destructure(m2);
    c1, c2 =uniform(θ1,θ2)
    return re1(c1), re2(c2)
end

function gaussian_mlp(σ::Real = 1.0)
    vop = gaussian(σ)
    function mutation(recombinant::T) where {T <: Chain}        
        θ, re = Flux.destructure(recombinant)
        return re(convert(Vector{Float32}, vop(θ)))
    end
    return mutation
end

#the initial_population function of evolutionary function
function initial_population(method::M, individual::Chain) where {M<:Evolutionary.AbstractOptimizer}
    θ, re = Flux.destructure(individual);
    [re(randn(length(θ))) for i in 1:Evolutionary.population_size(method)]
end
#Overload some function for evolutionary function
NonDifferentiable(f, x::Chain) = NonDifferentiable{Real,typeof(x)}(f, f(x), deepcopy(x), [0,])

copy(ch::Chain) = deepcopy(ch)
# copy weight and bias between two models
function copyto!(layer1::Dense{T}, layer2::Dense{T}) where {T}
    copyto!(layer1.W, layer2.W)
    copyto!(layer1.b, layer2.b)
    return l1
end

function copyto!(ch1::Chain, ch2::Chain)
    for i in 1:length(ch1.layers)
        copyto!(ch1.layers[i],ch2.layers[i])
    end
    return ch1
end

# the iterations is performed later
opts = Evolutionary.Options(iterations=1, show_every=10, show_trace=false, store_trace=false)
algo = GA(
        selection = rouletteinv,
        mutation =  gaussian_mlp(1.5),
        crossover = uniform_mlp,
        # the following params needs to be tuned for the better results
        mutationRate = 0.99, # mutation rate too small will cause the model to be stuck in local minimum, too large will cause the model unstable
        crossoverRate = 0.95,
        populationSize = 100,# change population_size will increase training time, but may increase the accuracy
        ε = 0.000002
    )

epochs = 500 # the number of epochs
# change population_size will increase training time, but may increase the accuracy
save_resultTest = zeros(epochs);
save_resultTrain = zeros(epochs);
l1 = Dense(19, 2*19, sigmoid)
l2 = Dense(2*19, 2)
model = Chain(l1, l2)#define the model 
#the softmax is removed because it does not need to train

for i in 1:10
    
    # load the models saved from HW1
    loadpath = string("models/model",i,".bson")
    @load loadpath weights
    Flux.loadparams!(model, weights)

    res = Evolutionary.optimize(fitness, model, algo, opts)
    evomodel = Evolutionary.minimizer(res)

    for j in 1:epochs
        res = Evolutionary.optimize(fitness, model, algo, opts)
        model= Evolutionary.minimizer(res)
        if j%(epochs/5) == 0
        @printf("Loss in expirment %d epoch: %d in test data is %f\n",i, j, loss(test_data,model))
        end
        save_resultTest[j] = save_resultTest[j] + accuracy(test_data, model)
        save_resultTrain[j] = save_resultTrain[j] + accuracy(train_data, model)
    end
end


save_resultTest = save_resultTest ./ 10;
save_resultTrain = save_resultTrain ./ 10;

plot(log.(1:epochs), save_resultTest,label="Test")
plot!(log.(1:epochs), save_resultTrain,label = "Train", title = "Accuracy", legend = :outertopleft)


simplified_x_train_result = []
for i  = 1:size(x_train,2)
    if softmax(model(x_train[:,i]))[1]>0.5
        push!(simplified_x_train_result, 0)
    else
        push!(simplified_x_train_result, 1)
    end
end
simplified_y_train = []
for i  = 1:size(x_train,2)
    if y_train[i] == 0
        push!(simplified_y_train, 0)
    else
        push!(simplified_y_train, 1)
    end
end

# ConfusionMatrix for the training data
print("ConfusionMatrix for the training data\n")
ConfusionMatrix()(simplified_x_train_result, simplified_y_train)


simplified_x_test_result = []
for i  = 1:size(x_test,2) 
    if softmax(model(x_test[:,i]))[1]>0.5
        push!(simplified_x_test_result, 0)
    else
        push!(simplified_x_test_result, 1)
    end
end
simplified_y_test = []
for i  = 1:size(x_test,2) 
    if y_test[i] == 0
        push!(simplified_y_test, 0)
    else
        push!(simplified_y_test, 1)
    end
end

# ConfusionMatrix for the test data
print("ConfusionMatrix for the test data\n")
ConfusionMatrix()(simplified_x_test_result, simplified_y_test)