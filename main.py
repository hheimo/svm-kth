import numpy,random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt


##Setting seed for random
random.seed(1234)

##Generating test data
classA = numpy.concatenate((numpy.random.randn(10, 2)*0.2+[1.5, 0.5],
                            numpy.random.randn(10, 2)*0.2 + [-1.5, 0.5]))
classB = numpy.random.randn(20, 2)*0.2 +[0, -0.5]

inputs = numpy.concatenate((classA, classB))

targets = numpy.concatenate((numpy.ones(classA.shape[0]),
                             -numpy.ones(classB.shape[0])))

N = inputs.shape[0] ## Number of rows ie. samples

##Randomly order data
permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


##Plotting the data
plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b. ')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r. ')

plt.axis('equal')
plt.show()


##Dual problem
##def objective():



##Calling minimize function
##ret = minimize()