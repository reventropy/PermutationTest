import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import scipy.stats as st


def permtest(g1,g2,alpha):

	def returnRHS(perms,pool):
		rhs=[]
		for p in perms:
			mask = np.in1d(pool,p)
			rhs.append(pool[np.where(~mask)[0]])
		return np.array(rhs)


	g1 = g1[~np.isnan(g1)]
	g2 = g2[~np.isnan(g2)]
	if (g1.shape[0]<2) or (g2.shape[0]<2):
		return np.nan,np.nan,np.nan


	#calculate p-value
	z=st.norm.ppf(1-(alpha)/2) #calculate this based on alpha
	g1l = g1.shape[0]
	g2l = g2.shape[0]
	tlen = g1l+g2l
	pool = np.concatenate([g1,g2])
	delta = np.abs(np.mean(g1) - np.mean(g2))
	allperms = list(combinations(pool,g1.shape[0]))
	N = len(allperms)
	rhs = returnRHS(allperms,pool)

	successes = 0.0
	for p in range(N):
		if np.abs(np.mean(allperms[p])-np.mean(rhs[p]))>=delta:
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
	g1 = vals[i,0:3]
	g2 = vals[i,3:6]
	pval,lb,ub = permtest(g1,g2,alpha)
	if pval<0.05:
		print i,pval,lb,ub





