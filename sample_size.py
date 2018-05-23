import numpy as np
import pandas as pd
import scipy as sp 
import json
import os



mfNameMap = json.load(open("oml_metafeature_map.json", "r"))

variables = {k:[] for k in mfNameMap.keys()}

dirpath = "test/metalearn/metafeatures/openmlComparisons/"
for root,dirs,files in os.walk(dirpath):
	for file in files:
		comparison = json.load(open(dirpath+file))
		shared = comparison["INCONSISTENT SHARED METAFEATURES"] + comparison["CONSISTENT SHARED METAFEATURES"]
		#print(shared)
		for x in shared:
			if "Numeric" not in list(x.keys())[0]:
				mf = list(x.keys())[0]
				diff = x[mf]["Difference"]
				variables[mf].append(diff)

mf_means = {k:np.mean(variables[k]) for k in variables.keys()}

mf_stdev = {k:np.std(variables[k]) for k in variables.keys() if len(variables[k]) is not 0}

sample_size = {k:((1.65*mf_stdev[k]/.05)**2) for k in mf_stdev.keys()}

mean = np.mean(list(sample_size.values()))
maximum = max(list(sample_size.values()))
median = np.median(list(sample_size.values()))
Q1 = np.percentile(list(sample_size.values()), 25)
Q3 = np.percentile( list(sample_size.values()), 75)
IQR = 1.5 * (Q3 - Q1)
upper = Q3 + IQR
lower = Q1 - IQR

print("BEFORE:")
print("Size: " + str(len(sample_size)))
print("max: " + str(maximum))
print("mean: " + str(mean))
print("median: " + str(median))
print("Q1: " + str(Q1))
print("Q3: " + str(Q3))
print("IQR: " + str(IQR))
print("Upper Limit: " + str(upper))
print("Lower Limit: " + str(lower))
print()

processed = {k:v for k,v in sample_size.items() if v < upper and v > lower}
for k,v in sample_size.items():
	print(str(v) + ": " + k)
print()

pmean = np.mean(list(processed.values()))
pmaximum = max(list(processed.values()))
pmedian = np.median(list(processed.values()))
pQ1 = np.percentile(list(processed.values()), 25)
pQ3 = np.percentile( list(processed.values()), 75)
pIQR = 1.5 * (pQ3 - pQ1)
pupper = pQ3 + pIQR
plower = pQ1 - pIQR

print("BEFORE:")
print("Size: " + str(len(processed)))
print("max: " + str(pmaximum))
print("mean: " + str(pmean))
print("median: " + str(pmedian))
print("Q1: " + str(pQ1))
print("Q3: " + str(pQ3))
print("IQR: " + str(pIQR))
print("Upper Limit: " + str(pupper))
print("Lower Limit: " + str(plower))
print()