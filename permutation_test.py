import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations
import scipy.stats as st


def permtest(g1,g2,alpha):

	#calculate p-value
	z=st.norm.ppf(1-(alpha)/2) #calculate this based on alpha
	g1l = len(g1)
	g2l = len(g2)
	tlen = g1l+g2l
	delta = np.abs(np.mean(g1) - np.mean(g2))
	allperms = list(permutations(g1+g2,tlen))
	N = len(allperms)
	successes = 0.0
	for p in allperms:
		s1 = p[0:g1l]
		s2 = p[g1l:tlen]
		if np.abs(np.mean(s1)-np.mean(s2))>=delta:
			successes+=1.0
	prop_failed=N-successes
	pval = successes/N
	#calculate CI using Binomial proportion CI, Wilson score interval with continuity correction

	T1=2*N*pval
	T2=(z*z)
	T3=z*np.sqrt((z*z)-(1/N)+(4*N*pval*(1-pval))+((4*pval)-2))+1
	T4=z*np.sqrt((z*z)-(1/N)+(4*N*pval*(1-pval))-((4*pval)-2))+1
	divisor = 2*(N+(z*z))
	lb = max(0.0,(T1+T2-T3)/divisor)
	ub = min(1.0,(T1+T2+T4)/divisor)
	return pval,lb,ub
	

alpha = 0.05
vals = pd.read_csv('testdata.csv').get_values()
for i in range(vals.shape[0]):
	g1 = list(vals[i,0:3])
	g2 = list(vals[i,3:6])
	pval,lb,ub = permtest(g1,g2,alpha)
	if pval<0.05:
		print i,pval,lb,ub





