import numpy as np 
import sys as sys

def neural_network(training_file,test_file,layers,units_per_layer,rounds):
    class_name = {}
    current_round = 1 # Guideline 4; if this is greater than rounds stop 

    # Converts last string to digit in the training file  
    def convert_to_numbers_training_file(_nparray):
        counter = 1
        # Finding the name of various classes 
        for i in range(len(_nparray)):
            if _nparray[i][-1] in class_name:
                continue
            else:
                class_name[_nparray[i][-1]] = counter
                counter+=1 
        # Converting the class name to numbers 
        for i in range(len(_nparray)):
            _nparray[i][-1] = class_name[_nparray[i][-1]]
        
        # Converting the entire array as a float 
        _nparray = _nparray.astype(float)
        return _nparray

    # Convert to numbers test file
    def convert_to_numbers_test_file(_nparray):
        for i in range(len(_nparray)):
            _nparray[i][-1] = class_name[_nparray[i][-1]]
        _nparray = _nparray.astype(float)
        return _nparray

    # To find the maximum absolute of training file 
    def find_maximum_absolute(_nparray):
        max = 0 
        for i in range(len(_nparray)):
            for j in range(len(_nparray[0])-1): # Avoiding the last term 
                if abs(_nparray[i][j]) > max:
                    max = abs(_nparray[i][j])
        return max

    # Normalizing the training and test data 
    def normalize_data(_nparray,max_val):
        for i in range(len(_nparray)):
            for j in range(len(_nparray[0])-1): # Avoiding the last term 
                _nparray[i][j] = _nparray[i][j] / max_val
        return _nparray

    # Generates a random value between -0.05 and 0.056 in unifrom distribution
    def gen_ran(): # Guideline 2 
        return np.random.uniform(-0.05,0.05)

    # Generates a new learning rate 
    def learning_rate(): # Guideline 3
        return np.power(0.98,current_round-1)
    
    # Calculates sum from previous level 
    def calculate_sum_from_previous_level(z,weight,curr_layer,index):
        sum = 0 
        for i in range(len(z[curr_layer-1])):
            sum += weight[curr_layer-1][i][index] * z[curr_layer-1][i]
        return sum

    # Calculates sigmoid functional value of provided unit 
    def calculate_sigmoid_value(unit):
        return np.divide(1,(1+np.exp(-unit)))

    def calculate_sum_from_forward_gradient(z,gradient,weight,layer,level):
        sum = 0
        for i in range(len(z[layer+1])):
            for k in range(len(z[layer])): 
                sum += gradient[layer+1][i] * weight[layer][level][i]
        return sum

    train = np.loadtxt(training_file,dtype=str)
    test = np.loadtxt(test_file,dtype=str)
    train = convert_to_numbers_training_file(train) # Will be the training file 
    test = convert_to_numbers_test_file(test) # Will be the testing file 
    maximum_absolute_training = find_maximum_absolute(train) # Guideline 1 
    maximum_absolute_test = find_maximum_absolute(test)
    train = normalize_data(train,maximum_absolute_training)
    test = normalize_data(test,maximum_absolute_test)
    # At this point we have all the required information converted (Double checked)

    # Details needed calulated here 

    dimension = len(train[0]) - 1 # Since we don't need the last column; 
    # dimension is basically the no of data input for the input layer 
    
    # Creating all required layers 
    z = [None]*layers 
    a = [None]*layers
    
    # Creating the first layer (Input Layer)
    z[0] = [None] * dimension
    a[0] = [None] * dimension

    # Creating the remaining hidden layers 
    for i in range(1,layers-1):
        z[i] = [None] * units_per_layer
        a[i] = [None] * units_per_layer

    # For the last output layer 
    z[-1] = [None] * len(class_name)
    a[-1] = [None] * len(class_name)
    # We have created a neural network 
    
    # Creating weights and bias for the entire neural network 
    bias = [None] * layers
    weight = [None] * layers

    # Creating bias for the entire network 
    for i in range(layers):
        bias[i] = [None] * len(z[i])
        for k in range(len(z[i])):
            bias[i][k] = gen_ran()
    # Creating weight for the entire network 
    for i in range(layers-1):
        weight[i] = [None] * len(z[i])
        for k in range(len(z[i])):
            weight[i][k] = [None] * len(z[i+1])
            for n in range(len(z[i+1])):
                weight[i][k][n] = gen_ran()
    # The last element on weight and first element on bias should be None 
    
    # target contains the values correct output value (Double checked)

    # Adding values starts here -- TODO -- 
    # Running n number of rounds
    for y in range(rounds):
        for o in range(len(train)):
                # These are the target values 
                
                target = [None] * len(class_name) # Might need to be fixed if needed 
                for w in range(len(target)): # Assigns 1 to the class that needs to be predicted if not zero is added 
                    if train[o][-1] == (w+1):
                        target[w] = 1 # Means falls in this class 
                    else:
                        target[w] = 0

                # Adding values z   
                for i in range(dimension):
                    z[0][i] = train[o][i]
                
                # Solid working till this point 

                # Creating new z[l][u] at each level 
                for l in range(1,layers):
                    for u in range(len(z[l])):
                        a[l][u] = bias[l][u] + calculate_sum_from_previous_level(z,weight,l,u)
                        z[l][u] = calculate_sigmoid_value(a[l][u])
                
                # For gradients 
                # Using the same details as z
                delta = [None] * len(z)
                for i in range(len(z)):
                    delta[i] = [None] * len(z[i])
                
                # Starting form the end 
                for u in range(len(delta[-1])):
                    delta[-1][u] = (z[-1][u]-target[u]) * z[-1][u] * (1-z[-1][u]) 

                # Updating hidden layers 
                for v in range(layers-2,0,-1):
                    for l in range(len(delta[v])):
                        delta[v][l] = calculate_sum_from_forward_gradient(z,delta,weight,v,l) * z[v][l] * (1-z[v][l])
                
                # Updating weights and biases 
                for l in range(1,layers):
                    for i in range(len(z[l])):
                        bias[l][i] = bias[l][i] - learning_rate()* delta[l][i]
                        for k in range(len(z[l-1])):
                            weight[l-1][k][i] = weight[l-1][k][i] - learning_rate() * delta[l][i] * z[l-1][k]
        current_round+=1
    # Checked till dis point 

    predicted_correctly = 0
    total = len(test) 
    # Classification 
    for o in range(len(test)):
        for k in  range(dimension):
            z[0][k] = train[o][k]
        to_predict = test[o][-1]
        for l in range(1,layers):
            for i in range(len(z[l])):
                    a[l][i] = calculate_sum_from_previous_level(z,weight,l,i)
                    z[l][i] = calculate_sigmoid_value(a[l][i])
        new_array = list(z[-1])
        max_val = max(new_array)
        max_index = new_array.index(max_val)
        class_identifier = {v: k for k, v in class_name.items()}
        accuraccy = 0
        if (max_index+1) == (to_predict):
            predicted_correctly+=1 
            accuraccy = 1
        print(f'ID={o+1}, \t predicted={class_identifier[max_index+1]}, \t true ={class_identifier[to_predict]}, \t accuracy={accuraccy:0}')
        
    print(f'Classification accuracy= \t {(predicted_correctly/total)}')        

neural_network(str(sys.argv[1]),str(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]))