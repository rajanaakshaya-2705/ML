import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\AKSHAYA\\Downloads\\enjoysport.csv")  # Adjust the path to your file
print("Dataset:\n", df)
l = np.array(df)[:, :-1]
print("\nAttributes:\n", l)
target = np.array(df)[:, -1]
print("\nTarget:\n", target)
n = len(target)
attr = len(l[0])  
h = None 
for i in range(n):
    if target[i] == 'yes':  
        h = l[i].copy() 
        break
if h is None:
    print("\nNo positive examples found in the dataset.")
else:
    print("\nInitial Hypothesis:", h)
    for i in range(i + 1, n):  # Start checking after the first positive example
        if target[i] == 'yes':  # Check if it's a positive example
            for k in range(attr):  # Compare each attribute
                if h[k] != l[i][k]:  # If attribute values differ
                    h[k] = '?'  # Generalize to '?'
    print("\nFinal Hypothesis:", h)
