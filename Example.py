# import the module
import Forester

## Iris Dataset Example - Start

# create a new instance of Forester with the iris dataset
iris = Forester.Forester(path="./data/iris.csv", train=True)

# create a single prediction example for the iris dataset
iris_val = [1,5.1,3.5,1.4,0.2]

# predict the class of the iris_val
iris_prediction = iris.make_prediction(iris_val)

# print the prediction
print('Iris Prediction: ' + iris_prediction)

# delete the instance of iris to free up memory
del iris

## Iris Dataset Example - End

## Heart Attack Dataset Example - Start

# create a new instance of Forester with the heart_attack dataset
heart_attack = Forester.Forester(path="./data/heart_attack.csv", train=True)

# create a single prediction example for the heart_attack dataset
heart_attack_val = [47,1,0,110,275,0,0,118,1,1,1,1,2]

# predict the class of the heart_attack_val
heart_attack_prediction = heart_attack.make_prediction(heart_attack_val)

# print the prediction
print('Heart Attack Prediction: ', heart_attack_prediction)

# delete the instance of heart_attack to free up memory
del heart_attack

## Heart Attack Dataset Example - End

## Mobile Price Dataset Example - Start

# create a new instance of Forester with the mobile_price dataset
mobile_price = Forester.Forester(path="./data/mobile_price.csv", train=True)

# create a single prediction example for the mobile_price dataset
mobile_val = [842,0,2.2,0,1,0,7,0.6,188,2,2,20,756,2549,9,7,19,0,0,1]

# predict the class of the mobile_val
mobile_prediction = mobile_price.make_prediction(mobile_val)

# print the prediction
print('Mobile Price Prediction: ', mobile_prediction)

## Mobile Price Dataset Example - End