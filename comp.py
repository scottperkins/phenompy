import csv
import numpy as np
import matplotlib.pyplot as plt


laptop = []
desktop = []
with open('testing_laptop.txt','r') as f:
    reader = csv.reader(f)
    for line in reader:
        for item in line:
            laptop.append(float(item))
with open('testing.txt','r') as f:
    reader = csv.reader(f)
    for line in reader:
        for item in line:
            desktop.append(float(item))
result =[]
for num in np.arange(len(laptop)):
    print((laptop[num]-desktop[num])/desktop[num])
    result.append( (laptop[num]-desktop[num])/desktop[num])
x=np.linspace(1,len(laptop),len(laptop))
plt.plot(x,result)
plt.savefig("verification.pdf")
plt.close()
