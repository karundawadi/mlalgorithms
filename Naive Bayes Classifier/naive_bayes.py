import numpy as np
import sys

def naive_bayes(training_file, test_file):
    training_data = np.loadtxt(training_file)
    dict_defined = {}
    
    # To make training data dynamic 
    size_attr = len(training_data[0]) - 1

    labels = [] # Hosts the values of labels 
    for i in range(len(training_data)):
        if training_data[i][-1] in dict_defined:
            dict_defined[training_data[i][-1]].append(training_data[i][0:size_attr].flatten())
        else:
            labels.append(training_data[i][-1])
            dict_defined[training_data[i][-1]] = [training_data[i][0:size_attr].flatten()]
    
    labels.sort()
    
    # Making a dynamic array for labels storage and their respective statistics 
    class_attribute_description = [] 
    
    for i in range(len(labels)):
        complete_column = []
        new_array = []
        for k in range(size_attr):
            for j in range(len(dict_defined[labels[i]])):
                complete_column.append(dict_defined[labels[i]][j][k])
            # TODO : Check if the standard deviation matches ;; see question requirements 

            standard_deviation = np.std(complete_column,ddof=1)

            if standard_deviation < 0.01 : 
                standard_deviation = 0.01 # Given on question 

            mean_calculated = np.mean(complete_column)
            new_array.append([mean_calculated,standard_deviation]) # This is the mean and std calculated with the help of numpy of the entire attribute array 
            complete_column = [] # This is to revert back the previous changes
        class_attribute_description.append(new_array)

    # Calculating p(x)
    probability_individual = [0] * len(labels)
    total_elements = 0

    for i in range(len(labels)):
        total_elements += len(dict_defined[labels[i]])

    for i in range(len(probability_individual)):
        probability_individual[i] = (len(dict_defined[labels[i]])/total_elements)

    # At this point probability_individual is the probability of each class and class_attribute_description has the necessary 
    # mean and standard deviation 

    # So far we have :
    # probability of Ck, probability of P(x|ck) - can be calculated using the formula above , p(x)
    
    # Training part is complete 

    # Printing the data values stored inside class_attribute_description
    for i in range(len(labels)):
        for j in range(size_attr):
            print("Class "+ str(labels[i].astype(int)) + ", attribute " + str(j+1)+", mean = "+str(np.around(class_attribute_description[i][j][0],2))+", std = "+str(np.around(class_attribute_description[i][j][1],2)))

    # Opening the test_file
    test_data = np.loadtxt(test_file)
    accuracy = 0
    for i in range(len(test_data)):
        associated_array = test_data[i]
        real_class = associated_array[-1].astype(int)
        test_values = associated_array[:-1]
        each_class_gausian = [1] * len(labels)
        for j in range(len(test_values)):
            overall_summation = 1
            for n in range(len(labels)):
                value_of_gausian = ((1/((class_attribute_description[n][j][1])*(np.sqrt(2*np.pi))))* (np.exp(
                                                - ((np.power((test_values[j]-class_attribute_description[n][j][0]),2)))*(1/((2*np.power(class_attribute_description[n][j][1],2))))
                                            )))
                each_class_gausian[n] *= value_of_gausian
        
        total_sum = 0 # Value of P(x)
        for l in range(len(each_class_gausian)):
            total_sum += (each_class_gausian[l]*probability_individual[l])
        
        new_overall_value = [0] * len(labels)
        for m in range(len(labels)):
            new_overall_value[m] = (each_class_gausian[m]*probability_individual[m])/total_sum

        highest = 0
        index = 0
        for z in range(len(labels)):
            if new_overall_value[z] > highest:
                highest = new_overall_value[z]
                index = z
        
        # This finds the sum of all 
        

        accuracy_new = 0.00
        if labels[index] == real_class:
            accuracy_new = 1.00
            accuracy += 1 
        print("ID= "+str(i+1)+", predicted= "+ str(labels[index].astype(int)) + ",probability = "+ str(np.around(highest,4))+", true="+str(real_class)+", accuracy="+str(accuracy_new))
    print("Classification accuracy="+str(np.around(accuracy/len(test_data),4)))

naive_bayes(str(sys.argv[1]),str(sys.argv[2]))