import matplotlib.pyplot as plt
import numpy as ny
from Valida import Ppl 
file = open('People.txt','r')
data = file.readlines()

print(data)
summary = {True : 0, False : 0}
labels = ["Ruleset Valid", "Ruleset Invalid"]
for record in data:
    newRec = record.split(" ")
    pplobj = Ppl(newRec[0],newRec[1],newRec[2],newRec[3],newRec[4])
    result = pplobj.rules()
    summary[result] += 1
    print(newRec, " ` ", result)
print("Total no. of records in dataset:",(summary[True]+summary[False]))
print("No. of records that validate the ruleset:", summary[True])
print("No. of records that didn't validate the ruleset:",summary[False])
myexp = [0.1,0.1]
Xdata = ny.array([summary[True],summary[False]])
plt.suptitle("Data Mining Practical 1")
plt.pie(Xdata,labels = labels,explode = myexp, shadow = True)
plt.show()