import collections
from pprint import pprint
import pyquil.quil as pq
from pyquil.gates import *
import pyquil.api as api
import numpy as np
import random
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

qvm = api.SyncConnection()

numIter = 100
N_Q = 6
sCount = [0]*numIter
binStr = tuple([0]*N_Q)
toHist = []
toHistBin = []
for _iter in range(numIter):
    ins = pq.Program()
    # step 1
    for i in range(N_Q):
        ins.inst(H(i))

    #step 2
    # a)

    d = 40
    gates = [RX, RY, T]
    for k in range(d):
        pairings = np.arange(N_Q,dtype=int)
        np.random.shuffle(pairings)
        for l in range(int(N_Q/2)):
            p0 = int(pairings[2*l])
            p1 = int(pairings[2*l+1])
            ins.inst(CZ(p0, p1))
    # b)
        
        for qubit in range(N_Q):
            gate = random.choice(gates)
            if gate is T:
                ins.inst(gate(qubit))
            else:
                ins.inst(gate(np.pi/2, qubit))

    for q in range(N_Q):
        ins.measure(q, q)
    # print(ins)
    numTrials = 1000
    samples = qvm.run(ins, range(N_Q), trials=numTrials)
    sCount[_iter] = dict();

    for s in samples:
        sKey = tuple(s)
        sCount[_iter][sKey] = sCount[_iter].get(sKey, 0)
        sCount[_iter][sKey] += 1
    for key in sCount[_iter]:
        sCount[_iter][key] /= (numTrials + 0.0) 
        sCount[_iter][key] *= 2**N_Q
        toHist += [sCount[_iter][key]]
    toHistBin += [sCount[_iter].get(binStr, 0.0)]
    print("------------------------ iteration: {}, sCount:{}".format(_iter, sCount[_iter]))

 


 #Histogram of the Porter-Thomas Distribution 

#histogram of the dataset
n, bins, patches = plt.hist(toHist, 100, normed=1, facecolor='green', alpha=0.75)

#plot features 
plt.xlabel('Dp')
plt.ylabel('Pr[Dp]')
plt.title(r'$\mathrm{Histogram\ of\ Dp}$')
plt.axis([0, 7, 0, 1.6])
plt.grid(True)
plt.savefig('histPT.pdf', format='pdf', dpi=1000)
plt.show()



# Binning the Data 
binSize = 0.5
bins = [i*binSize for i in range(int(np.ceil(10/binSize)))]

f1 = np.digitize(toHist, bins)
f1Counts = collections.Counter(f1)

Xlist = []
Ylist = []
for i in range(len(f1Counts)):
    Xlist.append((i+1)*0.25)
    Ylist.append(f1Counts[i])

Yax = np.array(Ylist[1:])/sum(Ylist[1:]) 
Xax = np.array(Xlist[1:]) 



#Log-Lin Plot of the Prob[Dp] // Porter-Thomas

plt.plot(Xax, Yax, linewidth=3)
plt.axvline(x=1, ymin=0, ymax=1, color='green', linewidth=2, linestyle='dashed')
plt.xlabel('Dp')
plt.ylabel('Prob[Dp]')
#plt.title(r'$\mathrm{Plot\ of\ Probability \ of \ Dp:}$')
plt.axis([0.5, 4.2, 1/10**4, 1])
plt.grid(True)
plt.yscale('log')
plt.savefig('loglinPT.pdf', format='pdf', dpi=1000)
plt.show()
