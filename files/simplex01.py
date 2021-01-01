"""
A procedure illustrating an animated SIMPLE 3D Self Organizing Maps (SOMs) solution:
    - ALGORITHM:
        - in this instance, create (3 by 3 by 4 net) by (22 features/variables vectors)
        - fill the net with initial guesses (random numbers)
        - normalize the data coming in by columns
        - randomly grab an Species and find where it fits best (using absolute distances in this case)
        - adjust all values depending on distance, influence, and learning
        - run thru process again... and again...
    - OUTPUT
        - animated progress graphs illustrating membership in clusters
            - using top 3 features for color animation
            - reporting top 4 attribute animation
            - reporting counts in final report
        - membership lists based on SOM
        - list of features used in clustering
"""
import sys
import time
from pydataset import data
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib import gridspec


def find_absbm3(target, net):
    """
    Self Organizing Maps
    """
    absbm_id = np.array([0, 0, 0])
    nFeatures = net.shape[3]
    min_dist = sys.maxsize
    # run thru 3 dimensions
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            for z in range(net.shape[2]):
                # pick a vector
                trial = net[x, y, z, :].reshape(nFeatures, 1, 1)
                # see how far away the two vectors are using absolute values
                abs_dist = np.sum(abs(trial - target))  ###  large difference
                if abs_dist < min_dist:
                    min_dist = abs_dist
                    absbm_id = np.array([x, y, z])
    absbm = net[absbm_id[0], absbm_id[1], absbm_id[2], :].reshape(nFeatures, 1, 1)
    return (absbm, absbm_id)

def generateReport(net,lstVars):
    """
    this is the reporter for Self Organizing Maps
        - report occurs at the end of all iterations
        - routine for identifying and reporting membership of Species in clusters
        - returns the count of Species in each cluster
    """
    print("net.shape: ", net.shape)
    nFeatures = net.shape[3]
    print("*************  membership of Species ********************")
    cntGps = np.zeros([nXs,nYs,nZs],dtype=np.int)
    print("lstSpecies: \n", lstSpecies)
    print("lstSpecies[2]: ", lstSpecies[2])
    print("nFeatures: ", nFeatures)
    vec=[]
    for iRow in range(nRows):
        train = data[:, iRow].reshape(np.array([nFeatures, 1, 1]))
        absbm, absbm_id = find_absbm3(train, net)
        print(iRow, np.array(absbm_id)+1, lstSpecies[iRow])  ## 
        vec.append([iRow,absbm_id, absbm.T])
        iiX = absbm_id[0]
        iiY = absbm_id[1]
        iiZ = absbm_id[2]
        cntGps[iiX,iiY,iiZ]+=1
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            for z in range(net.shape[2]):
                train = net[x, y, z, :].reshape(np.array([nFeatures, 1, 1]))
                iRow+=1
                absbm_id = str([x,y,z]).replace(',','')
                absbm = train
                vec.append([iRow,absbm_id, absbm.T])
    print("*************  Features of Species  ********************")
    print("Features: \n", lstVars)
    unique, counts = np.unique(([str(vec[i][1]) for i in range(len(vec))]), return_counts=True)
    cntList = list(zip(unique,counts))
    cnt = [cntList[i][1] -1 for i in range(len(cntList))]
    return cntGps

def generatePlot(iIter,net,cntGps,nColx):
    """
    this is the plotter for Self Organizing Maps
        - after every 1000 iterations (in this case) plots the status of the net
        - figures out which are the 3 most important features based on the estimated Self Organizing Map
        - takes each vector of the 3 by 3 by 4 net and figures out its color based on the 3 most important features
        - creates a rectangle for each element of the net, colors and annotates the rectangle
        - in the final plot, includes the count of the number of Species in each cluster
    """
    patch = []
    nFeatures = net.shape[3]
    nPtMax = 5
    nColorMax = 3
    nColorPts = min(nRows,nColorMax)
    orderFace = np.argsort(np.abs(net.std(axis=(0,1,2))))[::-1][:nPlants]
    for x in range(1, net.shape[0] + 1):
        for y in range(1, net.shape[1] + 1):
            for z in range(1, net.shape[2] + 1):
                face = net[x-1,y-1,z-1,:]
                faceX = []
                for i in range(nColorPts):
                    faceX.append(face[orderFace[i]])
                varX = '1st>>'+str(orderFace[0])+": "+lstVars[orderFace[0]]+'    2nd>>'+str(orderFace[1])+": "+lstVars[orderFace[1]]+'    3rd>>'+str(orderFace[2])+": "+lstVars[orderFace[2]]+'    4th>>'+str(orderFace[3])+": "+lstVars[orderFace[3]]
                axes.annotate(varX,xy=(0.05,0.015),xycoords='axes fraction',fontsize=9,fontweight='normal')
                rect = plt.Rectangle(((x-1.0)/3, -0.13+(1.24*(y-0.68)+z*0.29)/4), 1.0/3.0, 0.25/3.0, facecolor=faceX,edgecolor='gray')
                patch.append(axes.add_patch(rect))
                face = [int(10000*face[i])/10000.0 for i in range(nFeatures)]
                strFace = ""
                for i in range(nColx):
                    strFace+=(str(orderFace[i])+":  "+str(face[orderFace[i]])+'\n')
                strXYZ = "["+str(x)+","+str(y)+","+str(z)+"]"
                if iIter >= nCnt-1:
                    cntGp = cntGps[x-1][y-1][z-1]
                    strXYZ+= ": ("+str(cntGp)+")"
                strFace+=strXYZ
                #-----------------------------------------------------
                colorX = 'orange'
                if faceX[0]>0.7 or faceX[1]>0.6 or faceX[2]>0.75:
                    colorX = 'black'
                axes.annotate(strFace,xy=((x-0.78)/(nXs+0.20),-0.015+0.98*(y-1.05+z*0.24)/(nYs+0.16)),xycoords='axes fraction',
                                 fontsize=7, color=colorX, fontweight='bold')
    return patch

def getData():
    iris = data('iris')
    lstSpecies = iris.Species.to_list()
    lstVars = list(iris.columns)[0:4]
    tuples = [tuple(x) for x in iris.values[0:,0:4]]
    raw_data = np.transpose(tuples)
    print("raw_data.shape: ", raw_data.shape)
    return (lstVars, lstSpecies, raw_data)

def animate(iIter):
    """"
            - radius:  how far do we search
            - learn:  how much are we able to learn
            - train:  transformed real data that trains the net
            - trial:  net vector that keeps iterating closer and closer to strong clusters
            - influence:  the closer other vectors are, the more they are influenced
            - learn:  the more often the cycles occur, the less there is to learn and learning rate drops
            - diff = trial - train:  the bigger the diff between the train and the trial data the bigger the adjustment required
            - net effect is summarized as:  (trial + (learn * influence * (train - trial)))
    """
    global net, axes, nCnt, nRowsPerIter, qInit
    global lstSpecies, nYs, nXs, nZs
    nRowsThisIter = nRowsPerIter
    if qInit:
        nRowsThisIter = 1
    qInit = False
    nFeatures = net.shape[3]
    nColx = nFeatures    #  how many figures to annotate
    if nColx > 4:
        nColx = 4
    initRadiusM1 = init_radius - 1
    for iX in range(nRowsThisIter):
        # identify a random Species
        intRnd = np.random.randint(0, nPlants)
        # pull the vector for that Species
        train = data[:, intRnd].reshape(np.array([nFeatures, 1, 1]))
        # find closest location to properties of that Species
        absbm, absbm_id = find_absbm3(train, net)
        # figure out how far into the process we have gone
        iStep = nRowsThisIter*iIter+iX
        # adjust the radius for how far we have advanced
        # calculate radius of search
        radix = 1 + initRadiusM1 * (n_iterations - iStep)/n_iterations   ### small difference
        # find square of radius for re-use
        radi_radi = radix * radix
        # adjust learning rate based on how far we have advanced
        learn = ipt2_learning_rate + (ipt1_learning_rate - ipt2_learning_rate)*(1.0 - iStep / n_iterations)   ### medium difference
        # run thru 3 dimensions
        for x in range(net.shape[0]):
            for y in range(net.shape[1]):
                for z in range(net.shape[2]):
                    #  adjust those close or closer to target based on distance away
                    wt_dist = np.sum((np.array([x, y, z]) - absbm_id) ** 2)
                    if wt_dist <= radi_radi:
                        trial = net[x, y, z, :].reshape(nFeatures, 1, 1)
                        # influence depends on how far away another vector is
                        influence = 1.0 - 0.5 * (wt_dist/(radi_radi))    ### small difference
                        net[x, y, z, :] = (trial + (learn * influence * (train - trial))).reshape(1, 1, nFeatures)
    axes.clear()
    axes.annotate(('iter: %d out of %d     top %d out of %d Features displayed' % (nRowsPerIter*(iIter+1),nRowsPerIter*nCnt,nColx,nFeatures)),
                     xy=(0.05,0.985),xycoords='axes fraction',fontsize=8,fontweight='normal')
    lstCntCluster = []

    cntGps = []
    if iIter >= nCnt-1:
        cntGps = generateReport(net,lstVars)

    patch = generatePlot(iIter,net,cntGps,nColx)

    if iIter >= nCnt-1:
        end_time = int(round(time.time() * 1000))
        diff_time = (end_time - start_time)/1000.0
        print (nCnt, "iterations in: ",diff_time," seconds")
 
    return patch

#  setting of random seed to insure test results are identical and testable
np.random.seed(123456)

start_time = int(round(time.time() * 1000))
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
               
iIter = 0
nRowsPerIter = 1000
n_iterations = 400000
# n_iterations = 2000

lstVars,lstSpecies,raw_data = getData()
print(">>>>>>>  raw data")
print("lstVars: ", lstVars)
print("lstSpecies: ", lstSpecies)
print("raw_data: ", raw_data)

nRows = raw_data.shape[1]
nCols = raw_data.shape[0]
nDims = 3
nYs = 3
nXs = 3
nZs = 4

net_dims = np.array([nYs, nXs, nZs])
nCnt = int(n_iterations/nRowsPerIter)
ipt1_learning_rate = 0.03
ipt2_learning_rate = 0.01
weight_range = [0, 1]
a = weight_range[0]
b = weight_range[1]

nFeatures = raw_data.shape[0]
nPlants = raw_data.shape[1]
init_radius = max(net_dims[0], net_dims[1], net_dims[2]) / 2
print("nFeatures, nPlants: ", nFeatures, nPlants)
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)
data = raw_data
print("************************ data ********************* \n", data)
col_maxes = raw_data.max(axis=1)
col_mines = raw_data.min(axis=1)
data = (raw_data - col_mines[:, np.newaxis]) / ((col_maxes-col_mines)[:, np.newaxis])
net = (b-a) * np.random.random((net_dims[0], net_dims[1], net_dims[2], nFeatures)) + a
print("^^^^^^^^^^^  net.shape: ", net.shape)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0]=10
fig_size[1]=9
plt.rcParams["figure.figsize"] = fig_size
fig = plt.figure()
gs = gridspec.GridSpec(1,1)
# setup axes
axes = fig.add_subplot(gs[0], aspect='equal')
axes.set_xlim((0.4, net.shape[0]+0.6))
axes.set_ylim((0.4, net.shape[1]+0.6))

plt.tight_layout()
qInit = True

ani = animation.FuncAnimation(fig, animate, nCnt, repeat=False, blit=False)
plt.show()




