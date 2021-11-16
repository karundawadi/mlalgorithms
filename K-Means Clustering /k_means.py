import numpy as np 
import sys 

# data_file is the path to the file
# k gives the number of clusters 
# initialization gives how to initialize the data points 
def k_means(data_file, k , initialization):
    assign_table = {
        "random": 0,
        "round_robin" : 1
    }
    previous_assignment = []
    
    # Adds 0 as the cluster classification 
    def add_cluster(data):
        x = data.ndim
        new_arr = []
        
        if x == 1:
            data = data[:,np.newaxis]
            
        for i in range(len(data)):
            val = list(data[i])
            val.append(0)
            new_arr.append(val)
            
        return np.array(new_arr)
    
    # Assigns in round robin fashion 
    def round_robin(data):
        choices = np.arange(1,k+1)
        counter = 0 
        current_choice = []
        for i in range(len(data)):
            if counter >= k:
                counter = 0
            current_choice.append(choices[counter])
            data[i][-1] = choices[counter]
            counter+=1
        nonlocal previous_assignment # Specifies this is a global variable 
        if np.array_equal(current_choice,previous_assignment):
            return (data, True) 
        else:
            previous_assignment = current_choice
            return (data, False)
    
    # Assigns values randomly 
    def random_cluster(data):
        for i in range(len(data)):
            data[i][-1] = np.random.randint(1,k+1)
        return data 
    
    # This adds to the corresponding clusters 
    def add_to_cluster(data):
        clusters = {}
        for i in range(len(data)):
            if data[i][-1] in clusters:
                clusters[data[i][-1]].append(list(data[i]))
            else:
                val = list(data[i])
                clusters[data[i][-1]] = [val]
        return clusters 
    
    def caluclate_mean(clusters,dimension):
        if dimension == 1:
            choices = np.arange(1,k+1)
            overall = []
            for i in choices: 
                mean = 0
                if i in clusters: # Sometimes i does not exist as this is random assignment 
                    for j in clusters[i]:
                        mean += j[0] # Since second index is a cluster 
                    overall.append((i,(mean/len(clusters[i]))))
            return overall
        else:
            choices = np.arange(1,k+1)
            overall = []
            for i in choices:
                mean_x = 0
                mean_y = 0
                if i in clusters:
                    for j in clusters[i]:
                        mean_x += j[0] # Since second index is a cluster 
                        mean_y += j[1]
                    overall.append((i,(mean_x/len(clusters[i])),(mean_y/len(clusters[i]))))
            return overall
    
    def calculate_l_2(x1,x2,dimension):
        if dimension == 1:
            return np.sqrt(np.power(x1-x2,2))
        else:
            return np.sqrt(np.power(x1[0]-x2[0],2)+np.power(x1[1]-x2[1],2))
    
    def reassign_clusters(data, mean, dimension):
        if dimension == 1:
            new_assignment = []
            for i in range(len(data)):
                lowest_val,index = 9999999999,0
                for j in range(len(mean)):
                    val, _mean = mean[j]
                    distance = calculate_l_2(data[i][0],_mean,dimension)
                    if distance < lowest_val:
                        lowest_val = distance
                        index = j
                _val, __mean = mean[index]
                new_assignment.append(_val)
                data[i][-1] = _val         
            nonlocal previous_assignment
            if np.array_equal(new_assignment,previous_assignment):
                return True 
            else:
                previous_assignment = new_assignment
                return False 
        else: # This means we are on 2 d array 
            new_assignment = []
            for i in range(len(data)):
                lowest_val,index = 9999999999,0
                for j in range(len(mean)):
                    val, _mean_x, _mean_y = mean[j]
                    distance = calculate_l_2([data[i][0],data[i][1]],[_mean_x,_mean_y],dimension)
                    if distance < lowest_val:
                        lowest_val = distance
                        index = j
                _val, _mean_x, _mean_y = mean[index]
                new_assignment.append(_val)
                data[i][-1] = _val         
            if np.array_equal(new_assignment,previous_assignment):
                return True 
            else:
                previous_assignment = new_assignment
            return False 
    
    data = np.loadtxt(data_file)
    dimension = data.ndim
    # Inserting the cluster labels 
    data = add_cluster(data) # Cluster is added 
    repeated = False
    init_choice = assign_table[initialization]
    
    # Assigning the values accordingly to the user's choice
    if init_choice == 0:
        data = random_cluster(data)
    elif init_choice == 1:
        data,repeated = round_robin(data)
    else:
        print("Wrong choice")
        exit(-1)
    clusters = add_to_cluster(data)
    check_repeated = False
    while not check_repeated:
        mean = caluclate_mean(clusters,dimension)
        check_repeated = reassign_clusters(data,mean,dimension)
        clusters = add_to_cluster(data)
    print()
    if dimension == 2:
        for i in range(len(data)):
            print(
                f'({data[i][0]:10.4f},\t{data[i][1]:10.4f}) ---> cluster {int(data[i][2])}')  
    else: # Since we are guarantted to have only two dimension 
        # This is the case for 1 d 
        for i in range(len(data)):
            print(
                f'{data[i][0]:10.4f} ---> cluster {int(data[i][1])}')  
    print()
    

k_means (str(sys.argv[1]),int(sys.argv[2]),str(sys.argv[3]))