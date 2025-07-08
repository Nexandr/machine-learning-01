import numpy as np

#a bunch of the code was taken from the previous single-variable-linear-regression non vectorized version file

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

#prediction using given theta values
def pred_fn(X, theta):
    return X @ theta

#scochastic gradient descent (idk how to use the vectors work)
def gradient_descent(X, Y, theta, alpha):
  pred=pred_fn(X,theta)
  error=pred-Y
  grad=(1/len(X))*(X.T @ error)
  theta-=alpha*grad
  return theta

#main where all the fun stuff happens
def main():
    print("Gradient Descent")
    print("----------------")

    X,Y=collect_values()
    
    #normalization
    mean_x=np.mean(X[:,1])
    std_x=np.std(X[:,1])
    X[:,1]=(X[:,1]-mean_x)/std_x
    
    if len(X)==0 or len(Y)==0:
        print("No values entered in X or Y.")
        return

    
    alpha = get_values_float("\nEnter learning rate:")
    iterations = int(get_values_float("\nEnter number of iterations:"))
    theta=np.zeros(2)
    for i in range(0,iterations):
        theta=gradient_descent(X, Y, theta, alpha)

    print("\nCalculated Parameters Are:")
    print("\n", theta)

    print("-----------------------\n")
    req_x=float(input("Enter value of X for which to predict value of Y:"))
    normalized_req_x=(req_x - mean_x)/std_x

    #new hypothesis
    calc_y = theta[0]+(normalized_req_x*theta[1])
    print("\nCalculated value of Y is:", calc_y)
    
main()
