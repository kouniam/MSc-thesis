import random
from pack_unpack import unpack_matrix

T = 0.15            # network temperature
Iex = 0             # external stimulation parameter
steps = 1000000     # simulation steps

print(f'Initial opinion average = {np.sum(Sij)/N}')

#local field calculations

#breaking the Aij matrix into local groups
#(breaking Sij needs to be done on the go, because it's constantly updated!)
uA = unpack_matrix(Aij,Lgx,Lgx)

Sp = np.zeros(steps)

for tt in range(steps):
   
    i = random.randrange(0,L)
    j = random.randrange(0,L)
            
    hij = 0
    iisum = 0
    losum = 0
    
    uS = unpack_matrix(Sij,Lgx,Lgx) #local groups of Sij

    neigh1 = list(G.neighbors((i,j)))
    for z in range(len(neigh1)):
        
        #index selection
        tup1 = neigh1[z]
        z1 = tup1[0]
        z2 = tup1[1]
        
        #contribution of interpersonal connections
        iisum += Aij[z1,z2]*Sij[z1,z2]
        
    #local group identification
    ix = np.floor(i/Lg)
    jx = np.floor(j/Lg)
    
    #contribution of local group
    losum = np.sum(uS[(ix, jx)]*uA[(ix, jx)])
        
    hij = (1/kij[i,j])*(iisum+(losum/Ng))+Iex

    #opinion switch mechanism
    if T == 0:
        
        if hij*Sij[i,j] >= 0:
        
            pij = 0
        
        else:
        
            pij = (1-Aij[i,j])
        
            #probability "dice roll"
            Proll = np.random.random()
        
            if Proll <= pij:
            
                Sij[i,j] = (-1)*Sij[i,j]
                
    else:
        
        if hij*Sij[i,j] > 0:
        
            pij = (1-Aij[i,j])*np.exp((-hij*Sij[i,j])/T)
            
            #probability "dice roll"
            Proll = np.random.random()
        
            if Proll <= pij:
            
                Sij[i,j] = (-1)*Sij[i,j]
        
        else:
            
            pij = (1-Aij[i,j])*(1-np.exp((hij*Sij[i,j])/T))
            
            #probability "dice roll"
            Proll = np.random.random()
        
            if Proll <= pij:
            
                Sij[i,j] = (-1)*Sij[i,j]

    
    Sp[tt] = np.sum(Sij)/N
    
    #print(f'Step = {tt} ; <S> = {np.sum(Sij)/N}')
    
    if tt % 10000 == 0:                    
        plt.figure(figsize=(7,5))
        plt.imshow(Sij, cmap='coolwarm')
        plt.axis('off')
        plt.savefig('frame_%d.png' % tt, format='png')
        plt.close()

print(f'Final opinion average={np.sum(Sij)/N}')
