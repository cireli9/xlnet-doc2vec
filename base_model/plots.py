import matplotlib.pyplot as plt

## Bag-size vs error
bag_sizes = list(range(1000, 9001, 1000))
train = [0.86772,0.89656,0.91584,0.92848,0.93824,0.94548,0.94992,0.95516,0.95992]
test = [0.85324,0.871,0.8772,0.88304,0.88492,0.88704,0.88848,0.88876,0.88976]
plt.plot(bag_sizes, train, 'r', bag_sizes, test, 'b')
plt.xlabel("Max # of Features")
plt.ylabel("Accuracy")
plt.title("Feature Count vs Accuracy for C = 0.05")
plt.legend(["Training Accuracy", "Testing Accuracy"])
plt.show()
