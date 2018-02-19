import numpy,random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ================================================== #
# Kernel Functions
# ================================================== #
def linearKernel(x, y):
    return numpy.dot(numpy.transpose(x), y) + 1

def quadraticKernel(x, y):
    return (numpy.dot(numpy.transpose(x), y)+1)**2

sigma = 4

def rbfKernel(x, y):
    d = np.substract(x, y)
    temp = -(numpy.dot(d, d)/(2*np.power(sigma, 2)))
    return numpy.exp(temp)


# ================================================== #
# Generating data
# ================================================== #

random.seed(3000)

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

# ================================================== #
# SVM
# ================================================== #


##Building kernel-matrix
kernel = numpy.zeros((N, N))

for i in range(0, N):
    for j in range(0, N):
        kernel[i, j] = linearKernel([(inputs[i])[0], (inputs[i])[1]],[(inputs[j])[0], (inputs[j])[1]])

P = numpy.outer(targets, targets)*kernel
Q = numpy.ones(N)*-1
H = numpy.zeros(N)


##Objective function
def objective(a):
    temp = 0
    for i in range(N):
        for j in range(N):
            temp += 0.5*a[i]*a[j]*P
    temp -= numpy.sum(a)
    return temp

def zerofun(a):
    for i in range(0, len(a)):
        if(a[i] < 0 or a[i] > C):
            return False
    if(numpy.dot(a, targets) != 0):
        return False



#Initial guess values
start = numpy.zeros(N)
B = [(0, None) for b in range(N)]
XC ={'type':'eq', 'fun':zerofun}
C = 5


ret = minimize(objective, start, bounds=B, constraints=XC)
alpha = ret['x']


# ================================================== #
# Plotting data
# ================================================== #

plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b. ')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r. ')

##Plotting decision boundary
xgrid = numpy.linspace(-5, 4)
ygrid = numpy.linspace(-4, 4)



plt.axis('equal')
plt.show()


##Dual problem
##def objective():



##Calling minimize function
##ret = minimize()