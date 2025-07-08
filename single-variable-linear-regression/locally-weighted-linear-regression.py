#yeah i reuse code from my previous coding projects

import numpy as np


def get_values_float(prompt):
    user_input=input(prompt)
    if user_input.lower()=='end':
        return None
    try:
        return float(user_input)
    except ValueError:
        print('Please enter a valid input or "end" to finish.\n')

#loop to collect values and returning them as numpy arrays
def collect_values():
    X_list=[]
    Y_list=[]
    counter=0
    print('\nEnter training data (enter "end" to finish.)\nCurrently entered =', counter)
    while len(X_list)<100:
        x_input=get_values_float(f"\nThere are currently {counter} values in X. Enter a value for next data pair:")
        if x_input is None:
            break
        y_input=get_values_float("Enter the corresponding value for Y:")
        if y_input is None:
            break
        X_list.append([1,x_input])
        Y_list.append(y_input)
        counter+=1
    X=np.array(X_list)
    Y=np.array(Y_list)
    print(f"Collected {counter} data pairs.")
    return X, Y

#locally weighted prediction
def locally_weighted_predict_fn(X, Y, norm_x, tau):
    
    weights=np.exp(-(((X[:,1]-norm_x)**2)/(2*(tau**2)))) #formula
    W_matrix=np.diag(weights) #converting into n*n matrix
    
    theta = np.linalg.pinv(X.T @ W_matrix @ X) @ (X.T @ W_matrix @ Y) #formula for theta (absolute hell pls let me free)
    req_input_vector = np.array([1, norm_x]) #turning input into a vector

    predicted_y = req_input_vector @ theta #output should by theta dot x thanks andrew ng
    return predicted_y
#pray to the almighty snake gods that this is comprehensible oh my days


#where all the fun stuff happens
def main():
    print("Locally Weighted Prediction")
    print("----------------")

    X,Y=collect_values()
    
    #normalization
    mean_x=np.mean(X[:,1])
    std_x=np.std(X[:,1])
    X[:,1]=(X[:,1]-mean_x)/std_x
    
    if len(X)==0 or len(Y)==0:
        print("No values entered in X or Y.")
        return

    tau = get_values_float("\nEnter the value of 'Tau':")

    print("-----------------------\n")
    req_x=float(input("Enter value of X for which to predict value of Y:"))
    normalized_req_x=(req_x - mean_x)/std_x

    #new hypothesis
    calc_y = locally_weighted_predict_fn(X, Y, normalized_req_x, tau)
    print("\nCalculated value of Y is:", calc_y)
    
main()
