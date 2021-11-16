import numpy as np 
import sys

# This is renamed to lamda instead of lambda due to python naming convention
def linear_regression(training_file, test_file, degree, lamda):
    # Will calculate the value of Phi based on degree 
    def calculate_phi(training_data, degree_provided):
        phi_before_transpose = []
        length_of_attribute = len(training_data) - 1
        for i in range(len(training_data)):
            attributes = training_data[i][:-1]
            to_append = [1]
            for j in range(len(attributes)):
                for k in range(degree_provided):
                    to_append.append(np.power(attributes[j],k+1))
            phi_before_transpose.append(to_append)
        phi_numpy_arr = np.asarray(phi_before_transpose) 
        np.transpose(phi_numpy_arr)
        return phi_numpy_arr
    
    # This is for test data only 
    def calculate_phi_test_data(test_array,degree_provided):
        attributes = test_array[:-1]
        to_append = [1]
        return_val = []
        for j in range(len(attributes)):
            for k in range(degree_provided):
                to_append.append(np.power(attributes[j],k+1))
        return_val.append(to_append)
        phi_numpy_arr = np.asarray(return_val) 
        return phi_numpy_arr.T
    
    # Will calculate the value of T based on file provided. (1 * n)
    def calculate_t(training_data):
        return_arr = []
        for i in range(len(training_data)):
            return_arr.append([training_data[i][-1]])
        return np.asarray(return_arr)

    # Training Part 
    training_data = np.loadtxt(training_file)
    len_of_attributes = len(training_data[0]) - 1 # 1 is subtracted to take care of the end class name 
    len_of_training_data = len(training_data)
    t = calculate_t(training_data) # This is all the sequence of data provided
    phi_with_degree = calculate_phi(training_data,degree)
    phi_dotted = np.dot(phi_with_degree.T,phi_with_degree)
    identity_matrix = np.identity(len(phi_dotted)) * lamda # Since we multiply the values we have a D * D matrix
    overall_array = identity_matrix + phi_dotted
    inversed_term = np.linalg.pinv(overall_array)
    w_training = np.dot(np.dot(inversed_term,np.transpose(phi_with_degree)),t)
    for i in range(len(w_training)):
        print("w"+str(i)+"="+str(np.around(w_training[i][0],4)))

    w_training_transposed = w_training.T
    
    # Testing Part 
    testing_data = np.loadtxt(test_file)
    len_of_attributes = len(testing_data) - 1
    len_of_training_data = len(testing_data)


    for i in range(len(testing_data)):
        phi_for_testing = calculate_phi_test_data(testing_data[i],degree)
        predicted_value = np.dot(w_training_transposed,phi_for_testing)[0][0]
        print(
            f'ID=\t {(i+1):3}, output=\t\t{np.around(predicted_value,4):1.4f}, target value =\t {np.around((testing_data[i][-1]),4):1.4f}, squared error = {np.around(np.power((testing_data[i][-1]-predicted_value),2),4):1.4f}' 
        )  

linear_regression(str(sys.argv[1]),str(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]))