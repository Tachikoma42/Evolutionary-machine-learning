# build PyCall
ENV["PYTHON"] = raw"C:\Users\xkzmx\anaconda3\python.exe"
using Pkg
Pkg.build("PyCall")

using PyCall
pd = pyimport("pandas")
skeLCS = pyimport("skeLCS")
sk1 = pyimport("sklearn.model_selection")
sk2 = pyimport("sklearn.metrics")


# load data and divided into train and test
data = pd.read_csv("nba_logreg.csv")
# predict target
classLabel = "TARGET_5Yrs"
dataFeatures = data.drop(classLabel,axis = 1).values
dataPhenotypes = data[classLabel].values
dataHeaders = data.drop(classLabel,axis=1).columns.values
x_train, x_test, y_train, y_test = sk1.train_test_split(dataFeatures, dataPhenotypes, random_state=0);

#Training model using training data
model = skeLCS.eLCS(learning_iterations = 50000,track_accuracy_while_fit=py"True");
trainedModel = model.fit(x_train,y_train)

# print Accuracy
trainScore = trainedModel.score(x_train,y_train)
#print("Final Training Accuracy: "+pystr(trainedModel.score(x_train,y_train)))
@info "Final Training Accuracy: "  trainScore

testScore = trainedModel.score(x_test,y_test)
@info "Final Test Accuracy: "  testScore

plot = pyimport("matplotlib.pyplot")
PyCall.fixqtpath()


#All three plots are displayed in popup windows.
filename = "iterationData.csv"
trainedModel.export_iteration_tracking_data(filename)
dataTracking = pd.read_csv(filename)

iterations = dataTracking["Iteration"].values
accuracy = dataTracking["Accuracy (approx)"].values

plot.xscale("log",base=10) 

plot.plot(iterations,accuracy,label="Accuracy")
plot.xlabel("Iteration")
plot.ylabel("Accuracy")
plot.legend()
plot.show()

sk2.plot_confusion_matrix(model, x_train, y_train)  
plot.show()

sk2.plot_confusion_matrix(model, x_test, y_test)  
plot.show()

#Display trained rules
py"""
def printdata(trainedModel,data, filename):
    classLabel = "TARGET_5Yrs"
    trainedModel.export_final_rule_population(data.drop(classLabel,axis=1).columns.values, classLabel,filename,DCAL=False)
"""

py"printdata"(trainedModel,data,"rules.csv")
populationData2 = pd.read_csv("rules.csv")