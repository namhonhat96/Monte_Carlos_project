import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os



flag = 0
for subdir, dirs, files in os.walk("output"):
	for i in range(8):
		path =os.path.join(subdir, "output"+str(i)+".txt")
		if (flag):
			joiner = pd.read_csv(path, sep=',', header=None)
			df = pd.merge(df,joiner,left_index=True,right_index=True,suffixes=(i-1,i))
		else:
			df = pd.read_csv(path, sep=',', header=None)
			flag = 1
			
print(df)

for index,row in df.iterrows():
    plt.plot(range(8),row)
plt.show()

print("AVG")
for column in df:
    print(df[column].mean())
print("VAR")
for column in df:
    print(df[column].var())
