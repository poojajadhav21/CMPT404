import math
import matplotlib.pyplot as plt
#Here 'n' is the initial sample size that we assume
n = 1000
#initializing given values in the problem 2.12
Dvc = 10
e = 0.05
d = 0.05
#initialize iter count to 0
iter = 0
Iterations = []
for i in range (1,10):
    #Calculating the sample size using the equation N > 8/(Epsilon^2)(ln(4((2N)^dvc+1)))/delta) and subsitituting the given values we get
	N = (8/(math.pow(e,2))) * math.log(4*(math.pow(2*n,Dvc)+1)/d)
	#Adds the N values at the end of the list
	Iterations.append(N)
	print "The values for N and n after interation", iter, ":"
	print "N :", N
	print "n :", n
	print 
	#increment the iteration count by 1
	iter = iter + 1
	if round(N) == n:
		print "Total number of iterations : ", iter	
		break
	else:
	#This will round off the double value to the nearest integer
		n = round(N)
		continue
print Iterations
plt.plot(Iterations)
plt.title('Sample Size N Vs Iterations')
plt.xlabel('Number of Iterations')
plt.ylabel('N Samples size')
plt.show(Iterations)