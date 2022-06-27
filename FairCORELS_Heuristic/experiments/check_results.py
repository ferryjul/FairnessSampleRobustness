from os import listdir
from os.path import isfile, join
import pandas as pd
import math

path = "./results-test"
path_old_expes = "../../../data"
onlyFiles = [f for f in listdir(path) if isfile(join(path, f))]

for aFile in onlyFiles:
    if aFile[-4:] == '.csv':
        #print(type(aFile)) # str
        #print(aFile) # files names
        content = pd.read_csv('%s/%s' %(path, aFile))
        content_old_expes =  pd.read_csv('%s/%s' %(path_old_expes, aFile))
        match = True
        for l in range(len(content.values)):
            for c in range(len(content.values[l])):
                if content.values[l][c] != content_old_expes.values[l][c]:
                    if not(math.isnan(content.values[l][c]) and math.isnan(content_old_expes.values[l][c])):
                        match = False
                        #print(content.values[l][c], " != ", content_old_expes.values[l][c])
                        break
        if not match:
            print("Results mismatch for file %s." %aFile)
