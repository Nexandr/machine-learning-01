import numpy as np
import matplotlib.pyplot as mat

#uhhh this does erh basically it checks whether the value the user has
#entered is float or not, if it is 'end' it returns none, if it is not
#end, then it tries to convert the input data to float - this will
#happen only if input data is float, double or int, else it will show
#the error

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
    predictions=[]
    for i in range(len(X)):
        predictions.append(theta[0]+(theta[1]*X[i, 1]))
    return np.array(predictions)

#scochastic gradient descent (idk how to use the vectors work)
def gradient_descent(X, Y, theta, alpha):
    predictions=pred_fn(X, theta)
    for i in range(len(X)):
        error = predictions[i]-Y[i]
        theta[0]=theta[0]-(alpha*error*X[i,0])
        theta[1]=theta[1]-(alpha*error*X[i,1])
    return theta

#main where all the fun stuff happens
def main():
    print("Gradient Descent")
    print("----------------")

    X,Y=collect_values()
    if len(X)==0 or len(Y)==0:
        print("No values entered in X or Y.")
        return

    
    alpha = get_values_float("\nEnter learning rate:")
    iterations = int(get_values_float("\nEnter number of iterations:"))
    theta=np.zeros(2)
    for i in range(0,iterations):
        theta_upd=gradient_descent(X, Y, theta, alpha)

    print("\nCalculated Parameters Are:")
    print("\n", theta_upd)

    print("-----------------------\n")
    req_x=float(input("Enter value of X for which to predict value of Y:"))

    #new hypothesis
    calc_y = theta_upd[0]+(req_x*theta_upd[1])
    print("\nCalculated value of Y is:", calc_y)

    
    #matpplotlib time woohooo
    mat.scatter(X[:, 1], Y, label='Training Data', color='black')
    x_plot = np.linspace(min(X[:, 1]) - 1, max(X[:, 1]) + 1, 100)
    y_plot = theta[0]+(theta[1]*x_plot)
    mat.plot(x_plot, y_plot, color='blue')
    mat.scatter([req_x], [calc_y], color='green', s=100, label='Prediction Value (%.4f)' % calc_y)
    mat.title("Linear Regression")
    mat.xlabel("X")
    mat.ylabel("Y")
    mat.legend()
    mat.grid(True)
    mat.show()
    
main()
