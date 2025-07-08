# UNIVARIATE REGRESSION MODELS
yea most of them are just univariate regression models ahaha yeah thats it


---

## Projects in This Repo

### 1. `linear_regression.py`
simple single variable linear regression model built through iterative loops because vectors are scary

### 2. `vectorized_linear_regression.py`
vectors aint so scary anymore are they (spent half of my braincells during debugging yes they definately are in python atleast)

### 3. `locally_weighted_linear_regression.py`
basically linear regression but instead of global parameters this calculates local parameters based upon the closeness of the position relative to queried x.

this was... an experience

---

## 🛠 What does the code do?

Takes a series of (x, y) input pairs from the user  
Trains a regression model using one of the three methods above  
Asks for a new input and predicts the corresponding output value  
In the LWR version, adjusts the prediction dynamically using a `tau` value (bandwidth)

---

## 📚 What libraries does it use?

- Just **NumPy**
- There's some **matplotlib** commented out somewhere... dont ask me about it

---
