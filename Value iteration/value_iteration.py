# Karun Dawadi 
# 1001660099

import numpy as np 
import sys 

def value_iteration(environment_file,non_terminal_reward, gamma,k):
    
    # Constant variables 
    termial_state_count = 0
    obstacle_count = 0
    total_valid_states = 0
    actions = { # Defines the set of valid actions 
        "left" : [-1,0],
        "right" : [1,0],
        "top" : [0,1],
        "bottom" : [0,-1]
    }
    environment = None
    
    # Converts the given file to an two dimensional environment array
    def read_file(environment_file):
        provided_file = open(environment_file,"r").read().splitlines()
        environment = []
        for line in provided_file:
            section = line.split(",")
            append_array = []
            nonlocal termial_state_count
            nonlocal obstacle_count
            nonlocal total_valid_states
            for individual in section: 
                if individual == "1.0": # Terminal State
                    append_array.append(1)
                    termial_state_count+=1
                    total_valid_states +=1 
                elif individual == "-1.0": # Terminal State
                    append_array.append(-1)
                    termial_state_count+=1
                    total_valid_states +=1 
                elif individual == '.':
                    append_array.append(non_terminal_reward)
                    total_valid_states +=1 
                elif individual == "X":
                    append_array.append(None)
                    obstacle_count+=1
                else:
                    pass
            environment.append(append_array)
        return environment
    
    # This actually prints the environmental two dimensional array 
    def print_environment(val):
        for i in range(len(val)):
            for j in range(len(val[0])):
                if val[i][j] == None:
                    print("None \t\t", end=" ")
                else:
                    print(f'{val[i][j]:6.3f}\t\t',end=" ")
            print()
    
    # This prints the policy states in accordance to the requirement 
    def print_policy(val):
        for i in range(len(val)):
            for j in range(len(val[0])):
                if val[i][j] == None:
                    print("X \t\t", end=" ")
                else:
                    print(f'{val[i][j]}\t\t',end=" ")
            print()
            
    # Calculates the reward of a state 
    def reward(state):
        nonlocal environment
        x,y = state
        return environment[x][y] # This might return None 
    
    # Checks for the validity of a state 
    def check_state_validity(state):
        x,y = state
        nonlocal environment
        if (x < len(environment) and y < len(environment[0])) and (x >= 0 and y >= 0): # Cause sometimes we are negative values also 
            if environment[x][y] == None:
                return None 
            else:
                return True 
        else:
            return False # Out of the grid 
    
    # Finds the maximum of actions 
    def find_max(_U,state):
        # We have four actions - Up, Down, Left, Right 
        actions_utility = {}
        x,y = state
        
        # Defining various conditions as per the slides 
        probability_right_direction = 0.8 # Can be change accordingly if needed 
        probability_perpendicular_right = 0.1
        probability_perpendicular_right = 0.1
        
        max = -999999 
        
        for j in actions:
            action_x,action_y = actions[j]
            positive_ninty_x, positive_ninty_y =  action_y, -action_x # Clockwise rotation is +ve 
            negative_ninty_x, negative_ninty_y   = -action_y, action_x # Anti-clockwise rotation
            
            # Do check for every condiiton and move forward 
            _x_action_x , _y_action_y = x+action_x, y+action_y
            _x_positive_ninty_x, _y_positive_ninty_y = x+positive_ninty_x, y+positive_ninty_y
            _x_negative_ninty_x , _y_negative_ninty_y = x+negative_ninty_x, y+negative_ninty_y
            
            # Checking validity of all the generated states
            if check_state_validity([_x_action_x,_y_action_y]) == None or check_state_validity([_x_action_x,_y_action_y]) == False :
                    # Means we cannot go there and need to return back to the original location 
                    _x_action_x,_y_action_y = x,y
            
            if check_state_validity([_x_positive_ninty_x,_y_positive_ninty_y]) == None or check_state_validity([_x_positive_ninty_x,_y_positive_ninty_y]) == False :
                    # Means we cannot go there and need to return back to the original location 
                    _x_positive_ninty_x,_y_positive_ninty_y = x,y
            
            if check_state_validity([_x_negative_ninty_x,_y_negative_ninty_y]) == None or check_state_validity([_x_negative_ninty_x,_y_negative_ninty_y]) == False :
                    # Means we cannot go there and need to return back to the original location 
                    _x_negative_ninty_x,_y_negative_ninty_y = x,y
            
            # 0.8 and other are the rewards for that state 
            utility_h_action = _U[_x_action_x][_y_action_y] * probability_right_direction
            utility_h_positive_ninty = _U[_x_positive_ninty_x][_y_positive_ninty_y] * probability_perpendicular_right
            utility_h_negative_ninty = _U[_x_negative_ninty_x][_y_negative_ninty_y] * probability_perpendicular_right
            
            # In case we do not have any coming back to the same state case 
            total_utility = utility_h_action + utility_h_positive_ninty + utility_h_negative_ninty
            
            if total_utility > max:
                max = total_utility
        return max
    
    # The actual algorithm to implement the algorithm; minor change with k as it is 
    # taken as the no of runs 
    def valueIteration():
        nonlocal environment
        nonlocal k 
        nonlocal gamma
        _U = []
        
        # Due to python shallow copy using different method 
        # Was using [0 * len(environment[0])] * len(environment)
        
        for i in range(len(environment)):
            append_val = []
            for j in range(len(environment[0])):
                append_val.append(0)
            _U.append(append_val)
        
        
        U = []
        while k > 1:
            for i in range(len(environment)):
                append_val = []
                for j in range(len(environment[0])):
                    append_val.append(_U[i][j])
                U.append(append_val)
        
            for i in range(len(_U)-1,-1,-1):
                for j in range(len(_U[0])):
                    if reward([i,j]) == None:
                        continue # For the condition with blocked space 
                    elif reward([i,j]) == -1:
                        _U[i][j] = -1
                    elif reward([i,j]) == 1:
                        _U[i][j] = 1
                    else:
                        _U[i][j] = reward([i,j]) + gamma * find_max(U,[i,j])
            U = []
            k-=1
        return _U               
    
    # This finds the best action side 
    def find_max_policy(_U,state):
        # We have four actions - Up, Down, Left, Right 
        actions_utility = {}
        x,y = state
        
        # Defining various conditions as per the slides 
        probability_right_direction = 0.8 # Can be change accordingly if needed 
        probability_perpendicular_right = 0.1
        probability_perpendicular_right = 0.1
        
        signs = { # Fixing signs according to the assignment 
        # However need to fix the actual naming later 
            "left" : "^",
            "right" : "v",
            "top" : ">",
            "bottom" : "<"
        } # Done using assignment details 
        
        max = -999999 
        sign_to_return = "o"
        for j in actions:
            action_x,action_y = actions[j]
            positive_ninty_x, positive_ninty_y =  action_y, -action_x # Clockwise rotation is +ve 
            negative_ninty_x, negative_ninty_y   = -action_y, action_x # Anti-clockwise rotation
            
            # Do check for every condiiton and move forward 
            _x_action_x , _y_action_y = x+action_x, y+action_y
            _x_positive_ninty_x, _y_positive_ninty_y = x+positive_ninty_x, y+positive_ninty_y
            _x_negative_ninty_x , _y_negative_ninty_y = x+negative_ninty_x, y+negative_ninty_y
            
            # Checking validity of all the generated states
            if check_state_validity([_x_action_x,_y_action_y]) == None or check_state_validity([_x_action_x,_y_action_y]) == False :
                    # Means we cannot go there and need to return back to the original location 
                    _x_action_x,_y_action_y = x,y
            
            if check_state_validity([_x_positive_ninty_x,_y_positive_ninty_y]) == None or check_state_validity([_x_positive_ninty_x,_y_positive_ninty_y]) == False :
                    # Means we cannot go there and need to return back to the original location 
                    _x_positive_ninty_x,_y_positive_ninty_y = x,y
            
            if check_state_validity([_x_negative_ninty_x,_y_negative_ninty_y]) == None or check_state_validity([_x_negative_ninty_x,_y_negative_ninty_y]) == False :
                    # Means we cannot go there and need to return back to the original location 
                    _x_negative_ninty_x,_y_negative_ninty_y = x,y
            
            # 0.8 and other are the rewards for that state 
            utility_h_action = _U[_x_action_x][_y_action_y] * probability_right_direction
            utility_h_positive_ninty = _U[_x_positive_ninty_x][_y_positive_ninty_y] * probability_perpendicular_right
            utility_h_negative_ninty = _U[_x_negative_ninty_x][_y_negative_ninty_y] * probability_perpendicular_right
            
            # In case we do not have any coming back to the same state case 
            total_utility = utility_h_action + utility_h_positive_ninty + utility_h_negative_ninty
            
            if total_utility > max:
                max = total_utility
                sign_to_return = signs[j]
                
        return sign_to_return
    
    # This calculates the best action based on provided policies 
    def calculatePolicy(optimal_utility):
        nonlocal environment
        optimal_poilicy = []
        
        # Initializing the values 
        for i in range(len(environment)):
            append_val = []
            for j in range(len(environment[0])):
                append_val.append(None)
            optimal_poilicy.append(append_val)

        for i in range(len(optimal_utility)-1,-1,-1):
            for j in range(len(optimal_utility[0])):
                if reward([i,j]) == None:
                    optimal_poilicy[i][j] = "X"
                elif reward([i,j]) == -1:
                    optimal_poilicy[i][j] = "o"
                elif reward([i,j]) == 1:
                    optimal_poilicy[i][j] = "o"
                else:
                    optimal_poilicy[i][j] = find_max_policy(optimal_utility,[i,j])
        
        return optimal_poilicy
        
    environment = read_file(environment_file)
    value_iterated_array = valueIteration()
    optimal_policy = calculatePolicy(value_iterated_array)
    print()
    print("utilities:")
    print_environment(value_iterated_array)
    print()
    print("policies")
    print_policy(optimal_policy)
    print()
    
# Example input would be python value_iteration.py "environment2.txt" -0.02 1 15
value_iteration(str(sys.argv[1]),float(sys.argv[2]),float(sys.argv[3]),int(sys.argv[4]))