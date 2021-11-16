import sys
import numpy as np 

def decision_tree(training_file,test_file,option,pruning_thr):
    class_name = {}
    inv_class ={}

    class Node:
        def __init__(self,attribute,threshold,treeID,nodeID,informationGain):
            self.threshold = threshold 
            self.attribute = attribute 
            self.treeID = treeID
            self.nodeID = nodeID
            self.informationGain = informationGain
            self.left = None 
            self.right = None 
    
    class LeafNode:
        def __init__(self,val,leaf_id,treeID):
            self.val = val
            self.nodeID = leaf_id
            self.treeID = treeID
            self.attribute = -1
            self.threshold = -1
            self.informationGain = 0
            self.left = None 
            self.right = None 
    
    def convert_to_numbers_training_file(_nparray):
        counter = 0
        for i in range(len(_nparray)):
            if _nparray[i][-1] in class_name:
                continue
            else:
                class_name[_nparray[i][-1]] = counter
                counter+=1 
        
        for i in range(len(_nparray)):
            _nparray[i][-1] = class_name[_nparray[i][-1]]
        _nparray = _nparray.astype(float)
        return _nparray

    def convert_to_numbers_test_file(_nparray):
        for i in range(len(_nparray)):
            _nparray[i][-1] = class_name[_nparray[i][-1]]
        _nparray = _nparray.astype(float)
        return _nparray
    
    # Converts option to numeric value 
    def convert_option(option):
        _dict = {
            "optimized":0,
            "randomized":1,
            "forest3":3,
            "forest15":15,
            "forest500":500
        }
        return _dict[option]

    # Only the end of the array is passed to this function
    def calculate_distribution(_array):
        distribution = [None] * len(class_name) # Since there are n number of class_name
        for i in class_name.values():
            if len(_array) > 0:
                distribution[i] = np.count_nonzero(_array == i) / len(_array)
            else:
                distribution[i] = 0
        return distribution

    # This returns true or false if the remaining values are same and return value
    def checkIfClassesAreSame(examples):
        remainingClasses = examples[:,-1:]
        remainingClasses = np.unique(remainingClasses)
        return (len(remainingClasses) == 1)

    # Calculates the entropy of the passed in array 
    def calculate_entropy(_array):
        distribution = calculate_distribution(_array)
        entropy = 0
        for i in distribution:
            if i == 0:
                entropy += 0 # To mitigate the converging to infinity
            else:
                entropy += -i*np.log2(i)
        return entropy

    # Calculates the information gain when threshold applied
    # Needs examples, attributes and a threshold provided 
    def information_gain(examples,i,threshold):
        parent_entropy = calculate_entropy(examples[:,-1])
        left_array,right_array = examples[examples[:,i] < threshold],examples[examples[:,i] >= threshold]
        if len(left_array) > 0: left_entropy = calculate_entropy(left_array[:,-1]) * (len(left_array)/len(examples))
        else: left_entropy = 0
        if len(right_array) > 0: right_entropy = calculate_entropy(right_array[:,-1]) * (len(right_array)/len(examples))
        else: right_entropy = 0
        return parent_entropy-left_entropy-right_entropy # This is the information gain

    def choose_attribute_optimized(examples,attributes):
        max_gain = best_attribute = best_threshold = -1 
        for i in range(attributes):
            attribute_values = examples[:,[i]] # Since we already changed this in begining function calls 
            L = min(attribute_values)
            M = max(attribute_values)
            for j in range(1,51):
                threshold = L + j*(M-L)/51
                gain = information_gain(examples,i,threshold)
                if gain>max_gain:
                    max_gain=gain
                    best_attribute=i
                    best_threshold=threshold
        return (best_attribute,best_threshold,max_gain)

    def choose_attribute_randomized(examples,attributes):
        max_gain = best_threshold = -1 
        A = np.random.randint(16)
        attribute_values = examples[:,[A]] # Selects the attributes vertically
        L = min(attribute_values)
        M = max(attribute_values)
        for j in range(1,51):
            threshold = L + j*(M-L)/51
            gain = information_gain(examples,A,threshold)
            if gain>max_gain:
                max_gain=gain
                best_threshold=threshold
        return (A,best_threshold,max_gain)

    def DTL_optimized(examples,attributes,default,pruning_thr,tree_index,option,_id):
        if len(examples) < pruning_thr:
            return LeafNode(default,_id,tree_index)
        elif checkIfClassesAreSame(examples):
            return LeafNode(calculate_distribution(examples[:,-1:]),_id,tree_index)
        else:
            (best_attribute,best_threshold,max_gain) = choose_attribute_optimized(examples,attributes)
            tree = Node(best_attribute,best_threshold,tree_index,_id,max_gain)
            _id +=1 # Updating the value of ID here 
            examples_left, examples_right = examples[examples[:,best_attribute] < best_threshold],examples[examples[:,best_attribute] >= best_threshold]
            dist = calculate_distribution(examples[:,[-1]])
            tree.left = DTL_optimized(examples_left,attributes,dist,pruning_thr,tree_index,option,2*_id)
            tree.right = DTL_optimized(examples_right,attributes,dist,pruning_thr,tree_index,option,2*_id+1)
            return tree 
    
    def DTL_randomized(examples,attributes,default,pruning_thr,tree_index,option,_id):
        if len(examples) < pruning_thr:
            return LeafNode(default,_id,tree_index)
        elif checkIfClassesAreSame(examples):
            return LeafNode(calculate_distribution(examples[:,-1:]),_id,tree_index)
        else:
            (best_attribute,best_threshold,max_gain) = choose_attribute_randomized(examples,attributes)
            tree = Node(best_attribute,best_threshold,tree_index,_id,max_gain)
            _id +=1 # Updating the value of ID here 
            examples_left, examples_right = examples[examples[:,best_attribute] < best_threshold],examples[examples[:,best_attribute] >= best_threshold]
            dist = calculate_distribution(examples[:,-1:])
            tree.left = DTL_randomized(examples_left,attributes,dist,pruning_thr,tree_index,option,2*_id)
            tree.right = DTL_randomized(examples_right,attributes,dist,pruning_thr,tree_index,option,2*_id+1)
            return tree 

    def DTL_TopLevel(examples,pruning_thr,option_value,tree_index):
        default = calculate_distribution(examples[:,-1:])
        if option_value == 0:
            return DTL_optimized(examples,len(examples[0])-1,default,pruning_thr,tree_index,option_value,1)
        else:
            return DTL_randomized(examples,len(examples[0])-1,default,pruning_thr,tree_index,option_value,1)

    # Prints the tree in BFS format 
    def print_BFS(tree):
        queue, val = [tree],[]
        while queue:
            node = queue.pop(0)
            if node:
                val.append(node)
                queue.append(node.left)
                queue.append(node.right)
        
        for i in val:
            threshold = i.threshold
            g = np.array([0])
            feature = i.attribute
            if feature == -1:pass
            else:
                feature = feature+1
            if (isinstance(threshold,type(g))):
                threshold = threshold[0]
            print(
            f'tree= {i.treeID},\t\tnode= {i.nodeID},\t\tfeature = {feature},\t\tthr = {threshold:.2f},\t\tgain = {i.informationGain:.2f}' 
        )  

    # This does the actual prediction 
    def predict(attributes,tree):
        while tree.attribute != -1 and tree.threshold != -1:
            if attributes[tree.attribute] < tree.threshold:
                tree = tree.left
            else:
                tree = tree.right 
        return tree.val

    def predict_classes(test,tree):
        overall_accuracy = 0 
        tie = 0
        endvalues = test[:,-1]
        for i in range(len(test)):
            to_predict = endvalues[i]
            distribution = predict(test[:,:-1][i],tree)
            index_value = np.where(distribution == np.max(distribution))
            if len(index_value[0]) > 1:
                tie+=1 
            predicted = 0
            if index_value[0][0] == to_predict:
                predicted = 1
                overall_accuracy+=1
            
            print(
            f'ID=\t {(i+1):3},\t\tpredicted=\t\t{inv_class[index_value[0][0]]},\t\ttrue =\t {inv_class[to_predict]}, accuracy = {predicted}' 
            )  
        print(f'classification accuracy = {((overall_accuracy)/len(test)):6.4f}')

    def predict_trees(no,train,test,prun):
        overall_array = []
        for i in range(no):
            _appedn_this = []
            tree=DTL_TopLevel(train,prun,1,i)
            print_BFS(tree)
            for z in range(len(test)):
                distribution = predict(test[:,:-1][z],tree)
                _appedn_this.append(distribution)
            overall_array.append(_appedn_this)

        all_distribution = overall_array.pop()
        all_distribution = np.array(all_distribution)
        while overall_array:
            arr = np.array(overall_array.pop())
            for i in range(len(arr)):
                all_distribution[i] = all_distribution[i] + arr[i]
        for i in range(len(all_distribution)):
            all_distribution[i] = np.divide(all_distribution[i],no) # To average the values 
        
        overall_accuracy = 0
        for i in range(len(test)):
            distribution = all_distribution[i]
            to_predict = test[i][-1]
            index_value = np.where(distribution == np.max(distribution))
            predicted = 0
            if index_value[0][0] == to_predict:
                predicted = 1
                overall_accuracy+=1
            print(
            f'ID=\t {(i+1):3},\t\tpredicted=\t\t{inv_class[index_value[0][0]]},\t\ttrue =\t {inv_class[to_predict]}, accuracy = {predicted}' 
            )  
        print(f'classification accuracy = {((overall_accuracy)/len(test)):6.4f}')


    train = np.loadtxt(training_file,dtype=str)
    test = np.loadtxt(test_file,dtype=str)
    train = convert_to_numbers_training_file(train)
    test = convert_to_numbers_test_file(test)
    option_value = convert_option(option) 
    inv_class = {v: k for k, v in class_name.items()}
    if option_value == 0 or option_value == 1:
        tree = DTL_TopLevel(train,pruning_thr,option_value,1) # Pass in the tree index if this is a forest 
        print_BFS(tree)
        predict_classes(test,tree)
    else:
        trees = option_value
        predict_trees(trees,train,test,pruning_thr)
    return None 

decision_tree(str(sys.argv[1]),str(sys.argv[2]),str(sys.argv[3]),int(sys.argv[4]))