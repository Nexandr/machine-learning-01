import numpy as np

def get_values_float(prompt):
    while True:
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
    print('\nEnter training data (enter "end" to finish.)\nCurrently entered =', counter, 'Maxmimum datapoints allowed is 100.')
    while len(X_list)<100:
        x1_input=get_values_float(f"\nData point #{counter+1}, enter x\u2081:")
        if x1_input is None:
            break
        x2_input=get_values_float(f"Data point #{counter+1}, enter x\u2082:")
        if x2_input is None:
            break
        y_input=get_values_float("Enter the corresponding value for Y:")
        if y_input is None:
            break
        X_list.append([1,x1_input, x2_input])
        Y_list.append(y_input)
        counter+=1
    X=np.array(X_list)
    Y=np.array(Y_list)
    print(f"Collected {counter} data pairs.")
    return X, Y

#prediction using given theta values
def pred_fn(X, theta):
    return X @ theta

#vector GD
def gradient_descent(X, Y, theta, alpha):
  pred=pred_fn(X,theta)
  error=pred-Y
  grad=(1/len(X))*(X.T @ error)
  theta-=alpha*grad
  return theta

def pred_input(mean_x, std_x, theta):
    while True:
        try:
            req_x1=float(input("Enter value of x1 for which to predict value of Y:"))
            req_x2=float(input("Enter value of x2 for which to predict value of Y:"))
            break
        except ValueError:
            print('Incorrect Input Type. Please enter numeric values.')
    req_x_array = np.array([[req_x1, req_x2]])
    normalized_input = normalize(req_x_array, mean_x, std_x)
    req_x_biased = np.insert(normalized_input, 0, 1, axis=1)
    
    #new hypothesis
    calc_y = pred_fn(req_x_biased, theta)
    print(f"\nCalculated value of Y is: {calc_y[0]:.3f}")

#normalizing data
def normalize(features, mean_x, std_x):
    return (features - mean_x) / (std_x + 1e-8)

#main where all the fun stuff happens
def main():
    print("\n----------------\nGradient Descent\n----------------")

    X,Y=collect_values()
    if len(X)==0 or len(Y)==0:
        print("No values entered in X or Y.")
        return
    
    mean_x=np.mean(X[:,1:], axis=0)
    std_x=np.std(X[:,1:], axis=0)
    normalized_features = normalize(X[:,1:], mean_x, std_x)
    X[:,1:] = normalized_features

    print("-----------------------\n")
    while True:
        try:
            alpha=get_values_float('\nEnter the value of Learning Rate (Alpha):')
            if alpha is None:
                raise ValueError
            elif alpha<=0:
                print('Please make sure alpha is greater than 0.')
            else:
                break
            
        except ValueError:
            print('\nError: ValueError, please enter an valid numeric input for alpha')
    while True:
        try:
            iterations_float=get_values_float("\nEnter the number of iterations you'd like to perform:")
            if iterations_float is None:
                raise ValueError
            iterations=int(round(iterations_float))
            if iterations>=1e+9:
                print('Iteration count exceeds iteration cap at 1,000,000,000.')
            elif iterations<=0:
                print('Please make sure number of iterations is greater than 0.')
            else:
                print(f'Number of iterations: {iterations}\nValue of alpha: {alpha}')
                break
            
        except ValueError:
            print('\nError: ValueError, please enter an valid numeric input for iterations.')
            
    theta=np.zeros(3)
    for i in range(0,iterations):
        theta=gradient_descent(X, Y, theta, alpha)
    if not np.all(np.isfinite(theta)):
        print("\nGradient diverging. Consider using smaller alpha for useful iterations. (Diverging gradients can cause infinities or NaNs!)")
        return

    print("\nCalculated Hypothesis is:")
    #no this does not spawn multiple eldritch entities, it prints out the function
    print(f"\nH(x) = {f'{theta[0]:.2f}'.rstrip('0').rstrip('.')} + {f'{theta[1]:.2f}'.rstrip('0').rstrip('.')}\u22c5x\u2081 + {f'{theta[2]:.2f}'.rstrip('0').rstrip('.')}\u22c5x\u2082")
    print(f"The value of parameters are: {theta[0]:.8f}, {theta[1]:.8f}, {theta[2]:.8f}")
    while True:
        pred_key=input('-----------------------\nStart prediction?: (Yes/No or Y/N)')
        if pred_key.lower()=='yes' or pred_key.lower()=='y':
            pred_input(mean_x, std_x, theta)
        elif pred_key.lower()=='no' or pred_key.lower()=='n':
            break
        else:
            print('Please enter either "Yes/No" or "Y/N".')
            
run_count=0
while True:
    key = input('Run the program? (Yes/No or Y/N): ' if run_count == 0 else 'Run again? (Yes/No or Y/N): ')
    if key.strip().lower() in ['yes', 'y']:
        main()
        run_count += 1
    elif key.strip().lower() in ['no', 'n']:
        break
    else:
        print('Please enter either "Yes/No" or "Y/N".')

























































#i spend unhealthy amounts of time coding linear regression algorithms, its crazy how i haven't gone insane yet... or maybe i have idk
