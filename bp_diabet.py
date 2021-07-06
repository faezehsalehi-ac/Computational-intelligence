# Backprop on the Seeds Dataset
from random import seed 
from random import randrange
from random import random
from csv import reader
from math import exp

# Load a CSV file
def load_csv(filename):
	dataset = list()  # Define a liste for keeping dataset
	with open(filename, 'r') as file: # Open dataset file
		csv_reader = reader(file) # Read dataset file
		for row in csv_reader: # Load each rows of datas in dataset
			if not row: # if all rows finished break the if and continue the rest of code
				continue
			dataset.append(row) # append all rows to dataset that we defined it as list
	return dataset # return dataset list
 
# Convert string column to float
def str_column_to_float(dataset, column):
	for row in dataset: # for each rows in dataset 
		row[column] = float(row[column].strip()) #convert all values of each colomons to float>> ['6','1']>[6.0,1.0]
# Convert string column to integer (this function is used for dataset with some numeric features )
def str_column_to_int(dataset, column): 
	class_values = [row[column] for row in dataset] # creat a list to keep values of each colomn
	unique = set(class_values) # creat the set of class_arrays which means delete similar rows
	lookup = dict() # define a dictionary
	for i, value in enumerate(unique): # Put each value of uniqe set with their indexes together and keep them in lookup dictionary
		lookup[value] = i # assign numbers to each numeric value
	for row in dataset: # for each row in data set 
		row[column] = lookup[row[column]] # Replace all the numeric columns with produced numbers value 
	return lookup # return lookup dictionary

# Find the min and max values for each column
def dataset_minmax(dataset):
	minmax = list() # creat a list for keepping min and max values of each column
	stats = [[min(column), max(column)] for column in zip(*dataset)] #keep tupple min & max values of each columns in states
	return stats

# Rescale dataset columns to the range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset: #Input values vary in scale and need to be normalized to the range of 0 and 1   
		for i in range(len(row)-1):#the calculation to normalize a single value for a column is:
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]) #value-min/max-min

# Split a dataset into k folds -implement the k-fold cross-validation method
def cross_validation_split(dataset, n_folds):
	dataset_split = list() #creat a list
	dataset_copy = list(dataset) # copy of dataset to the above list
	fold_size = int(len(dataset) / n_folds) #foldsize=count(rows)/count(folds)
	for i in range(n_folds):#repeat 5 times
		fold = list()#create list for each fold
		while len(fold) < fold_size:#repeat fold-size
			index = randrange(len(dataset_copy))#to generate a random integer in the range between 0 and the size of the list
			fold.append(dataset_copy.pop(index))#creat foldsize of the records were assigned to the training dataset and remain for test set
		dataset_split.append(fold)
	return dataset_split #return list of train_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
	correct = 0#first time correct=0
	for i in range(len(actual)):#repeat until actual features finished : last column means 0=no diabet 1=has diabet
		if actual[i] == predicted[i]:#if our network predict true
			correct += 1#increase one to correct
	return correct / float(len(actual)) * 100.0#caculate percent of accuracy(confusion matrix)

# Evaluate an algorithm using a k-fold cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)#call cross_validation_split function
	scores = list()#scores is the list
	for fold in folds:#for every fold
		train_set = list(folds)#the dataset is split into train elements (all folds except fold that select)
		train_set.remove(fold)#delete selection fold from all folds
		train_set = sum(train_set, [])#set the train-set in the list
		test_set = list()#the dataset is split into test elements : test-set is the list
		for row in fold:#for all row in the fold
			row_copy = list(row)#make a copy of the test set in this line and next line
			test_set.append(row_copy)
			row_copy[-1] = None#each output value is cleared by setting it to the none value to prevent the algorithm from cheating accidentally
		predicted = algorithm(train_set, test_set, *args)#the algorithm provided as a parameter is a function tha expects the train and test datasets on which to prepare and then make predictiond.the algorithm  require the variable arguments *args in  the evalute-algorithm() function and passing them on to the algorithm function
		actual = [row[-1] for row in fold]#detect actual that use in prediction function: last row in the dataset
		accuracy = accuracy_metric(actual, predicted)#call accuracy-matrix function
		scores.append(accuracy)#keep accuracy(output of perivuos function in scores list
	return scores#return scores accuracy

# Calculate neuron activation for an input
def activate(weights, inputs):
	activation = weights[-1]#first time no active 
	for i in range(len(weights)-1):#for all neurons that have weights
		activation += weights[i] * inputs[i]#activation means weight*input
	return activation#return activation as the list

# Transfer neuron activation
def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))#sigmoid function : output between 0 ,1

# Forward propagate input to a network output
def forward_propagate(network, row):
	inputs = row #each row sells is input 
	for layer in network: #repeat for all layer in our network
		new_inputs = []#new-input is dictionary
		for neuron in layer:#for each neuron in layer
			activation = activate(neuron['weights'], inputs)#call activate function
			neuron['output'] = transfer(activation)#make output between 0 , 1 with sigmoid function
			new_inputs.append(neuron['output'])#keep previose outputs in new-inputs list
		inputs = new_inputs#save list in inputs 
	return inputs#return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):#derivative of sigmoid : use for acheive error
	return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs): # Start defining a network with number of inputs and hidden layers and number of output neurons
	n_layers = [n_inputs] + n_hidden + [n_outputs] # Callculate number of all layers which involves input layer, output layer and a bunch of hidden layers
	network = list() # Define our network as a list which involves neurons of each layers and their weights
	[network.append([{'weights':[random() for k in range(n_layers[i] + 1)]} for j in range(n_layers[i+1])]) for i in range(len(n_layers)-1)] # append all the weights( which were created randomlu) to the neurons of each layers
	return network # return our list 

# Make a prediction with a network
def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

# Test Backprop on Seeds dataset
seed(1)
# load and prepare data
filename = 'Diabet-1.csv'
dataset = load_csv(filename)
for i in range(len(dataset[0])-1):
	str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0])-1)
# normalize input variables
minmax = dataset_minmax(dataset)
normalize_dataset(dataset, minmax)
# evaluate algorithm
n_folds = 5
l_rate = 0.3
n_epoch = 500
n_hidden = list(map(int,input("Enter number of neurons in each hidden layer (with space): ").strip().split()))
scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
