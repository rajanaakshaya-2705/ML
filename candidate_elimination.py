import numpy as np
import pandas as pd
df = pd.read_csv("C:\\Users\\AKSHAYA\\Downloads\\enjoysport.csv")
print(df)
l = np.array(df)[:, :-1]
print("The attributes are: ", l)
target = np.array(df)[:, -1]
print("The target is: ", target)
s = l[0].copy() 
print("Initialization of specific hypothesis: ", s)
attr = len(s)
n = len(target)
g = [["?" for _ in range(attr)] for _ in range(attr)]
print("Initialization of general hypothesis: ", g)
for i in range(n):
    if target[i] == 'yes':
        print("If instance is Positive")
        for j in range(attr):
            if l[i][j] != s[j]:
                s[j] = '?' 
                g[j][j] = '?' 
    else:
        print("If instance is Negative")
        for j in range(attr):
            if l[i][j] != s[j]:
                g[j][j] = s[j]  
            else:
                g[j][j] = '?'  
    print("Step-", i)
    print("Specific Hypothesis: ", s)
    print("General Hypothesis: ", g)
gen = []
for i in g:
    if i.count('?') != attr:gen.append(i)
print("\nFinal Specific Hypothesis: ", s)
print("Final General Hypothesis: ", gen)
