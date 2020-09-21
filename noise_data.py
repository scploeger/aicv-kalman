#By: Spencer Ploeger
#Updated: May/31/2020
#create some sample data with noise
import numpy as np
import matplotlib.pyplot as plt
import csv

num_samples = 100

pure = np.linspace(-1, 1, num_samples) #num_samples spaced evenly between lower range and upper range
noise = np.random.normal(-0.05, 0.05, pure.shape) #generate some noise with the same shape as the "pure" signal
signal = pure + noise #add the noise to the signal

#print(signal)


#write the noisy signal to a file
with open('noisy.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(signal)

#write the "actual" signal to a file
with open('actual.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(pure)


plt.plot(signal)
plt.plot(pure)

plt.show()