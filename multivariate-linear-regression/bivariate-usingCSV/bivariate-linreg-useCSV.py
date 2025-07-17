import numpy as np
import pandas as pd

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
    while True:
        path = input("Enter path to .csv file (format: x1, x2, y):").strip()
        try:
            td = pd.read_csv(path)
            td.columns = td.columns.str.lower().str.strip()
            td.dropna(inplace=True)
            
            if not all(col in td.columns for col in ['x1', 'x2', 'y']):
                print('Given .csv file does not contain required column headers.')
                continue
            print(td.isnull().sum())
            X_ub = td[['x1', 'x2']].to_numpy(dtype=float)
            X = np.insert(X_ub, 0, 1, axis=1)
            Y = td['y'].to_numpy(dtype=float)
            return X, Y
        
        except FileNotFoundError:
            print("File not found. Check the path and try again.\n")
        except pd.errors.EmptyDataError:
            print("CSV is empty or badly formatted.\n")

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

def denormalize(theta, mean_x, std_x):
    denorm_theta = np.zeros(3)
    denorm_theta[1:] = theta[1:]/(std_x + 1e-8)
    denorm_theta[0] = theta[0] - (theta[1:]@(mean_x/(std_x + 1e-8)))
    return denorm_theta

def loss(X, Y, theta):
    predictions = pred_fn(X, theta)
    errors = predictions - Y
    loss = np.mean(errors**2)
    return loss

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
        if i % max(1,round(iterations/4)) == 0 or i == iterations-1:
            iteration_loss = loss(X, Y, theta)
            print(f"Loss in iteration #{i} is {iteration_loss:.2f}")
    print(f"Final Loss is: {loss(X, Y, theta):.4f}")
            
    if not np.all(np.isfinite(theta)):
        print("\nGradient diverging. Consider using smaller alpha for useful iterations. (Diverging gradients can cause infinities or NaNs!)")
        return

    print("\nCalculated Hypothesis is:")
    denorm_theta = denormalize(theta, mean_x, std_x)
    #no this does not spawn multiple eldritch entities, it prints out the function
    print(f"\nH(x) = {f'{denorm_theta[0]:.2f}'.rstrip('0').rstrip('.')} + {f'{denorm_theta[1]:.2f}'.rstrip('0').rstrip('.')}\u22c5x\u2081 + {f'{denorm_theta[2]:.2f}'.rstrip('0').rstrip('.')}\u22c5x\u2082")
    print(f"The value of parameters are: {denorm_theta[0]:.8f}, {denorm_theta[1]:.8f}, {denorm_theta[2]:.8f}")

    #MSE i learnt yesterday
    
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
