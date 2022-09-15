using Flux
using Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using Flux: @epochs
using MLJBase
using Printf
using BSON: @save # for save weights
using Plots

# get data and labels
images = Flux.Data.FashionMNIST.images() 
labels = Flux.Data.FashionMNIST.labels();
# the better way is to use MLDatasets, but it doesn't work on jupyter notebook.

# this is just to initialize, the 2 data is not used in the model
selectedImages = images[1:2];
selectedLabels = labels[1:2];

# filter the data so it contains only 0 and 1
for i in 1:60000
    if labels[i] == 0 || labels[i] == 1
        append!(selectedLabels, labels[i])
        push!(selectedImages, images[i])
    end
end

# get part of the selected data.
selectedTrainImage = selectedImages[3:1000]
selectedTrainLabel = selectedLabels[3:1000]
selectedTestImage = selectedImages[1001:1400]
selectedTestLabel = selectedLabels[1001:1400];

#reshape the data so it can be used in the model
x_train = hcat(float.(reshape.(selectedTrainImage, :))...) 
y_train = onehotbatch(selectedTrainLabel, 0:1)
x_test = hcat(float.(reshape.(selectedTestImage, :))...)
y_test = onehotbatch(selectedTestLabel, 0:1);

#define the model
#784->2*784->2
#output is the probability of being 0 or 1
model = Chain(Dense(28^2, 2*28^2, sigmoid), Dense(2*28^2, 2), softmax)
#@save "mymodel10.bson" model

#define the other functions that will be used in the model
loss(x, y) = crossentropy(model(x), y) 
optim = ADAM(); 
dataset = repeated((x_train,y_train),1); #in the first try, I tried to repeat the data to train, 
#but in the final version I change to loop the train process.
myaccuracy(x, y) = mean(onecold(model(x)) .== onecold(y))#modified the zoo example from julia


save_resultTest = []
save_resultTrain = []
epochs = 10 # the number of epochs


print("Start training\n")
for i = 1:epochs
    Flux.train!(loss, Flux.params(model), dataset, optim)
    @printf("Loss in epoch: %d in test is %f\n", i, loss(x_test, y_test))
    push!(save_resultTest, myaccuracy(x_test, y_test)) #save the accuracy of the test data
    push!(save_resultTrain, myaccuracy(x_train, y_train)) #save the accuracy of the training data
    #plot(log(i), loss(x_test, y_test))
end
print("Finish training\n")
# plot the accuracy in jupyter notebook
plot(log.(1:epochs), save_resultTest)
plot!(log.(1:epochs), save_resultTrain, title = "Accuracy")

# change the output from probability into  0 or 1
simplified_x_test_result = []
for i  = 1:400 
    if model(x_test[:,i])[1]>0.5
        push!(simplified_x_test_result, 0)
    else
        push!(simplified_x_test_result, 1)
    end
end
simplified_y_test = []
for i  = 1:400 
    if y_test[:,i][1] == true
        push!(simplified_y_test, 0)
    else
        push!(simplified_y_test, 1)
    end
end
# ConfusionMatrix for the test data
print("ConfusionMatrix for the test data\n")
ConfusionMatrix()(simplified_x_test_result, simplified_y_test )

simplified_x_train_result = []
# change the output from probability into  0 or 1
for i  = 1:998 
    if model(x_train[:,i])[1]>0.5
        push!(simplified_x_train_result, 0)
    else
        push!(simplified_x_train_result, 1)
    end
end
simplified_y_train = []
for i  = 1:998 
    if y_train[:,i][1] == true
        push!(simplified_y_train, 0)
    else
        push!(simplified_y_train, 1)
    end
end
# ConfusionMatrix for the training data
print("ConfusionMatrix for the training data\n")
ConfusionMatrix()(simplified_x_train_result, simplified_y_train)
