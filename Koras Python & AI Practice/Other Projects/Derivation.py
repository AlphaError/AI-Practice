
def runDerivative(myFunc, x): #h going to 0 makes it the approximate limit definition
    retX = (myFunc(x+0.0000000000001) - myFunc(x)) / 0.0000000000001
    return int(retX * 1.02) #bad fix
def squareFunc(x):
    return x*x #derivative(x^2, 4) => f(x)=2(x) of 4 == 8
def complexFunc(x):
    return x*x*x + 4*x*x + 2*x # x^3+4x^2+2x => 3x^2+8x+2
# print(runDerivative(squareFunc, 102.0))

# def findMinVal(myFunc, min, max):
#     if (max-min) <= 0.01:
#         return myFunc(min)
#     else:
#         times = (max - min) / 100.0
#         lowestVal = min
#         for x in range(times):
#             if myFunc(min+x) < myFunc(lowestVal):
#                 lowestVal = min + x
#         if(lowestVal == min):
#             return findMinVal(myFunc, lowestVal-10.0, lowestVal) #10 is just the step value
#         elif(lowestVal == max):
#             return findMinVal(myFunc, lowestVal, lowestVal+10.0)
#         else:
#             return myFunc(lowestVal)

def findMinVal(myFunc, min, max):
    lowestVal = myFunc(min)
    for x in range(min, max):
        if runDerivative(myFunc, x) < .0001 or runDerivative(myFunc, x) > -.0001:
            if myFunc(lowestVal) > myFunc(x):
                lowestVal = x
    return myFunc(lowestVal)

print(findMinVal(complexFunc, -5, 10)) #should be 0