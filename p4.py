import numpy as np
import matplotlib.pyplot as plt

def readin(filename):
    fi = open(filename,'r')
    matrix = np.zeros((100,100))
    i = 0
    for line in fi:
        line_str = line.split(',')
        matrix[i,:] = [ float(a) for a in line_str]
        i+=1
    fi.close()
    return matrix

def score(partition,matrix):
    #partiition is a (100,) array cotaining 0 and 1
    A_index = np.where(partition ==1)[0]
    B_index = np.where(partition ==0)[0]
    A_matrix = matrix[A_index,:]#[:,A_index]
    B_matrix = matrix[B_index,:]#[:,B_index]
    AB_matrix = matrix[A_index,:][:,B_index]
    A_mod = np.sum(A_matrix)
    B_mod = np.sum(B_matrix)
    AB_mod = np.sum(AB_matrix)
    score = 1.0e32
    if min(A_mod,B_mod) !=0: 
        score = AB_mod/min(A_mod,B_mod)
    return -1.0*score

def manyscore(parti_list,matrix):
    N = parti_list.shape[0]
    score_list = np.zeros(N)
    for i in range(N):
        score_list[i] = score(parti_list[i],matrix)
    return score_list

def sample(plist,N):
    u = np.random.uniform(0.0,1.0,100*N)
    u = np.reshape(u,(N,100))
    a = 1*(np.tile(plist,(N,1)) >u )  # p probablity to be in A and 1-p to be in B
    return a

def updateP(plist, sample, score_list, criteria,alpha):
    #sample (N,100) array meaning  N samples
    I = 1*(score_list>=criteria) 
    #print np.sum(I)
    pn = plist
    if np.sum(I) !=0:
        pn = np.sum(sample.transpose()*I ,axis=1)/np.sum(1.0*I)
    pnew = (alpha*pn) + (1.0-alpha)*plist
    return pnew

def updatebuff(plist,plist_buff):
    plist_buff[:-1,:] = plist_buff[1:,:]
    plist_buff[-1,:] = plist

def main():
    N = 200
    percentile = 0.02
    alpha = 0.2
    buff_size = 10
    #plist = np.random.uniform(0.0,1.0,100)
    plist = np.zeros(100)+0.5
    not_converged = True
    plist_buff = np.zeros((buff_size,100))
    plist_buff[-1] = plist
    matrix = readin("W_matrix.txt")
    gamma_highest = score(sample(plist,1)[0],matrix)
    gamma_buff = np.zeros(buff_size)
    it = 0
    while not_converged:
        it +=1
        sample_list = sample(plist_buff[-1],N)
        score_list = manyscore(sample_list,matrix)
        gamma = np.sort(score_list)[-int(N*percentile)-1]
        gamma_highest = max(gamma_highest,gamma)
        plistnew = updateP(plist_buff[-1],sample_list,score_list,gamma_highest,alpha)
        updatebuff(plistnew,plist_buff)
        gamma_buff[:-1] = gamma_buff[1:]
        gamma_buff[-1] = gamma
        #not_converged =  ((max(gamma_buff)-min(gamma_buff))>1.0e-16 )
        not_converged = (np.sum(np.var(plist_buff,axis=0)) > 1.0e-16 )
    print it
    plist_best = plist_buff[-1]
    #print plist_best
    partition = sample(plist_best,1)[0]
    score_best = -1.0*score(partition,matrix)
    return partition,score_best

scores = np.zeros(50)
partitions = np.zeros((50,100))
for i in range(50):
    partition,score_best = main()
    scores[i] = score_best

plt.plot(np.arange(50),scores,'o')
plt.show()
print(np.amin(scores))

#partition, score_best = main()
#print partition
#print score_best

