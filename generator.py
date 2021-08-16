import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_style('italic')

from scipy.stats import truncnorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datetime import datetime
startTime = datetime.now()

# MAIN MODEL PARAMETERS
L = 100             # size of the square grid
N = L**2            # number of individuals/nodes (L*L)
Lg = 20             # size of the local groups
Ng = Lg**2          # number of individuals/nodes per local group
Lgx = int(L/Lg)     # required for local group identifier

gamma = 3.0         # exponent of the power law distribution for the connectivity k
kmin = 10           # minimum number of connections of a node
kmax = 100          # maximum number of connections of a node
pc = 0.5            # probability to create second level connections

it = 2500           # number of iterations for the NETWORK generator

# acceptance counters

count_it = np.linspace(1, it, it, dtype = np.int32)

saturation = np.zeros(it)
avg_clustering = np.zeros(it)

count_primary = np.zeros(it)
count_secondary = np.zeros(it)
count_rejected = np.zeros(it)


# SECONDARY MODEL PARAMETERS

variance = 0.3  # sigma**2 (needed for truncated Gaussian distribution)

a = Lg      # parameter for distance probability function
b = Lg/4    # parameter for distance probability function

lcrit = 100  # critical distance parameter for node selection (improves speed)

# ------------------------------- #
# ---------- MAIN BODY ---------- #
# ------------------------------- #

# MATRIX GENERATORS

Sij = 2*np.random.randint(2, size=(L,L))-1    # generates the matrix Sij

# algorithm that ensures opinion average is 0 at the start (only works for even N)!
while np.sum(Sij)/N != 0:
    
    if (N % 2) != 0:
        break

    if np.sum(Sij)/N > 0:

        l1 = np.random.randint(0, L)
        l2 = np.random.randint(0, L)
    
        if Sij[l1,l2] > 0:
            
            Sij[l1,l2] = (-1)*Sij[l1,l2]
            
        else : continue
    
    else:
        
        l1 = np.random.randint(0, L)
        l2 = np.random.randint(0, L)
    
        if Sij[l1,l2] < 0:
            
            Sij[l1,l2] = (-1)*Sij[l1,l2]
            
        else: continue
    
print(f'Opinion average={np.sum(S)/N}')    

Kij = np.zeros(shape=(L,L))   #generator matrix for kij
count = -1                    #counting index

# connectivity matrix generated through a power law distribution
# re-maping function
def power_law(kmin, kmax, y, gamma):
    return ((kmax**(-gamma+1)-kmin**(-gamma+1))*y+kmin**(-gamma+1.0))**(1.0/(-gamma+1.0))

scale_free_dist = np.zeros(N, float)    # empty distribution matrix

# transforming a uniform distribution to a truncated power law shape
for n in range(N):
    scale_free_dist[n] = int(power_law(kmin, kmax,np.random.uniform(0,1), gamma))

# assignment of the distribution values to Kij matrix
for i in range(L):
    for j in range(L):
        count +=1
        Kij[i,j] = scale_free_dist[count]

# checking for the probability distribution function
kcount = np.zeros(101)

for n in range(N):
    ki = int(scale_free_dist[n])
    kcount[ki]+=1
    
# Gaussian distribution to generate authority matrix
mu, sigma = 0, np.sqrt(variance) # mean and standard deviation

#re-parametrized truncated normal distribution function
def get_truncated_normal(mean=0,sd=1,low=0,upp=10):
    return truncnorm(
            (low-mean)/sd,(upp-mean)/sd,loc=mean,scale=sd)
    
X = get_truncated_normal(mean=0,sd=sigma,low=0,upp=1)
    
A = X.rvs(N)    #authority values from distribution
    
Aij = np.zeros(shape=(L,L)) #authority matrix
count = -1  #counting index

#assignment of values to authority matrix
for i in range(L):
    for j in range(L):
        count +=1
        Aij[i,j] = A[count]

# saving the relevant arrays [optional] 
#np.save("S [L_%d;gamma_%d;pc_%f;LG_%d].npy" % (L,gamma,pc,Lg), Sij)
#np.save("K [L_%d;gamma_%d;pc_%f;LG_%d].npy" % (L,gamma,pc,Lg), Kij)
#np.save("A [L_%d;gamma_%d;pc_%f;LG_%d].npy" % (L,gamma,pc,Lg), Aij)

# (all matrix generator plots at the end [optional])

# NETWORK GENERATOR (slowest part)

kij = np.zeros(shape=(L,L)) # real number of connections of each node/individual
C = np.zeros(shape=(L,L))   # clusterring coefficient matrix

# generates a plus or minus sign with equal probability
def sign():
    return 1 if np.random.random()< 0.5 else -1

# creates a grid graph
G = nx.grid_2d_graph(L,L)

# position the nodes with equal distance in a lattice structure
pos = dict( (n, n) for n in G.nodes() )

# deletes all edges from the graph G, leaving only the nodes
G = nx.create_empty_copy(G)

# plots the empty graph (just to check) [optional]
#figG = plt.figure(figsize=(8,8))
#nx.draw(G,pos=pos,node_size=1,node_color='k',width=0.5)
#plt.axis('off')
#plt.savefig('empty_grid.pdf', format='pdf', bbox_inches='tight')
#plt.show()

# algorithm that generates second level connections
def second_order(i,j,n,m):

    # secondary connections (i,j) --> (n,m)
    neigh = list(G.neighbors((n,m)))            # list of neighbours of node (n,m)
    for z in range(len(neigh)):                 # iterate over all neighbours
       
        if (i,j) == neigh[z]:                   # ignores self: (i,j)
            continue

        if G.number_of_edges((i,j),neigh[z]) < 1:

            roll = np.random.random()
       
            if roll < pc:                             
       
                #index selection
                tupz = neigh[z]
                z1 = tupz[0]
                z2 = tupz[1]
                
                if kij[i,j] > kij[z1,z2]: 
                    count_rejected[tt] += 1
                    continue
       
                if kij[i,j] < Kij[i,j] and kij[z1,z2] < Kij[z1,z2]: 
                    G.add_edge((i,j),(z1,z2))   # adds a second level connection!
                    kij[i,j]+=1                                         
                    kij[z1,z2]+=1                                       
                    count_secondary[tt] += 1
                else:
                    count_rejected[tt] += 1
                    continue



# algorithm for creating interpersonal connections
def Interpersonal():
    for i in range(L):
        for j in range(L):
            
            if kij[i,j] == Kij[i,j]:  # checks if more connections are allowed
                count_rejected[tt] += 1
                continue
        
            # selects the random variables for indice pairing
            l1 = np.random.randint(0, lcrit)
            l2 = np.random.randint(0, lcrit)
        
            # selects the indices for the pairing
            n = i+sign()*l1
            m = j+sign()*l2
        
            # ensures that the index value is never out of the range [0,L]
            if n > L-1:
                n = n-L
            if m > L-1:
                m = m-L
            if n < 0:
                n = n+L
            if m < 0:
                m = m+L
                
            if i == n and j == m:   # in case it selects the same node, skip
                continue
        
            # distance between individuals
            l = np.sqrt((i-n)**2+(j-m)**2)
        
            # probability to form a connection between individuals P(l)       
            P = 1/(1+np.exp((l-a)/b))+0.001*((L-l)/L)
        
            # probability "dice roll"
            Proll = np.random.random()

            # connection generator algorithm now goes through all the conditional checks

            # first level connection between (i,j) and (n,m)
            if Proll < P:                                    
                if G.number_of_edges((i,j),(n,m)) < 1:      
                    if kij[i,j] > kij[n,m]:                
                        count_rejected[tt] += 1
                        continue
                    if kij[i,j] < Kij[i,j] and kij[n,m] < Kij[n,m]: 
                        G.add_edge((i,j),(n,m))   # adds a first level connection!
                        kij[i,j]+=1                                           
                        kij[n,m]+=1                                            
                        count_primary[tt] += 1
                    else: 
                        count_rejected[tt] += 1
                        continue
            
            # second level connections algorithm
            if G.number_of_edges((i,j),(n,m)) == 1:
                
                # secondary connections (n,m) --> (i,j)
                second_order(n,m,i,j)

                # secondary connections (i,j) --> (n,m)
                second_order(i,j,n,m)


# runs the network generator it times 
for tt in range(it):
    Interpersonal()
    
    # calculates saturation and clustering for each iteration [optional]
    # (computationaly expensive!)
    #saturation[tt] = np.sum(kij)/np.sum(Kij)
    #avg_clustering[tt] = nx.average_clustering(G)

    
print(f'Computation time = {datetime.now() - startTime}')
    
grad = Kij - kij     # visual check of the "quality" of the network

print(f'Network saturation = {np.sum(kij)/np.sum(Kij)}')    # network saturation
print(f'Network clustering = {nx.average_clustering(G)}')   # network clustering

count_accepted = count_primary + count_secondary

## GENERATOR PLOTS [OPTIONAL] ##
# color map of Sij matrix (1,-1)
fig, ax = plt.subplots(figsize=(8,8))
plt.title('Color map of Sij')
plt.xticks([])
plt.yticks([])
im = plt.imshow(Sij, cmap='coolwarm')

divider = make_axes_locatable(ax)
cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
fig.add_axes(cax)
fig.colorbar(im, cax=cax, orientation="horizontal")

plt.savefig('colormap_Sij_init.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# color map of Kij generator matrix (kmin,kmax)
fig, ax = plt.subplots(figsize=(8,8))
plt.title('Color map of Kij')
plt.xticks([])
plt.yticks([])
im = plt.imshow(Kij, cmap='YlOrBr')

divider = make_axes_locatable(ax)
cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
fig.add_axes(cax)
fig.colorbar(im, cax=cax, orientation="horizontal")

plt.savefig('colormap_Kij_gen.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

# color map of Aij  matrix (0,1)
fig, ax = plt.subplots(figsize=(8,8))
plt.title('Color map of Aij')
plt.xticks([])
plt.yticks([])
im = plt.imshow(Aij, cmap='YlGnBu')

divider = make_axes_locatable(ax)
cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
fig.add_axes(cax)
fig.colorbar(im, cax=cax, orientation="horizontal")

plt.savefig('colormap_Aij.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()


#Power law distribution
kx = np.round(np.linspace(kmin,kmax,kmax-kmin))

plt.figure(figsize=(7,5))
plt.title('Distribution of connectivity Kij')
plt.xlabel('k')
plt.ylabel('P(k)')
plt.yscale('log')
plt.xscale('log')
plt.plot(kx,kcount[kmin:kmax]/N,'r*')

plt.savefig('Kij_dist.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

xx = np.linspace(0,1,100)
plt.figure(figsize=(7,5))
plt.title('Distribution of authority Aij')
plt.xlabel('A')
plt.ylabel('P(A)')
plt.hist(A, 20, density=True)
plt.plot(xx, 2/(sigma * np.sqrt(2 * np.pi)) *
                np.exp( - (xx - mu)**2 / (2 * sigma**2) ),
          linewidth=1, color='r')
plt.show()
plt.close()

#Color map of Saturation
fig, ax = plt.subplots(figsize=(8,8))
plt.title('Saturation Kij-kij')
plt.xticks([])
plt.yticks([])
im = plt.imshow(grad)

divider = make_axes_locatable(ax)
cax = divider.new_vertical(size="5%", pad=0.5, pack_start=True)
fig.add_axes(cax)
fig.colorbar(im, cax=cax, orientation="horizontal")

plt.savefig('network_saturation.pdf', format='pdf', bbox_inches='tight')
plt.show()
plt.close()

plt.figure(figsize=(8,6))
plt.title('Acceptance')
plt.xlabel('Generator iterations',fontproperties=font)
plt.ylabel('Counts',fontproperties=font)
plt.yscale('log')
plt.xscale('log')
plt.grid(alpha=0.5)
plt.plot(count_it,count_rejected,'r-',label='Rejected')
plt.plot(count_it,count_accepted,'g-',label='Accepted (Total)')
plt.plot(count_it,count_primary,'c-',label='Accepted (Primary)')
plt.plot(count_it,count_secondary,'b-',label='Accepted (Secondary)')
plt.legend()

plt.savefig('acceptance.pdf', format='pdf', bbox_inches='tight')

plt.figure(figsize=(8,6))
plt.title('Saturation')
plt.xlabel('Generator iterations',fontproperties=font)
plt.ylabel('Saturation',fontproperties=font)
plt.grid(alpha=0.5)
plt.plot(count_it,saturation,'b-')

#np.save("count_it.npy", count_it)
#np.save("count_accepted.npy", count_accepted)
#np.save("count_rejected.npy", count_rejected)
#np.save("count_primary.npy", count_primary)
#np.save("count_secondary.npy", count_secondary)
