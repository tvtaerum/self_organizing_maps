
import time
import numpy as np
from matplotlib import pyplot as plt
# from matplotlib import patches as patches
import matplotlib.animation as animation
import pandas as pd
import sys
import operator
from matplotlib import gridspec

import matplotlib.pyplot as plt
import cartopy
import cartopy.io.shapereader as shpreader
import cartopy.crs as ccrs
import geopandas as gpd

def find_absbm3(id_target, target, net):
    """
    this is the brains of the Self Organizing Maps
        - passes in a real normalized data vector
        - identifies which target is closest to which vector (target vs trial)
        - returns the identifier and the vector
        - instead of using Euclidean distance it uses the sum of the absolute values as a measure of distance
        - any consistent measure of distance works
    """
    absbm_id = np.array([0, 0])
    nFeatures = net.shape[2]
    min_dist = sys.maxsize
    # run thru 2 dimensions
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            # pick a vector
            trial = net[x, y, :].reshape(nFeatures, 1)
            # see how far away the two vectors are using absolute values
            ### abs_dist = np.sum((w - t) ** 2)
            abs_dist = np.sum(abs(trial - target))  ###  large difference
            if abs_dist < min_dist:
                min_dist = abs_dist
                absbm_id = np.array([x, y])
    absbm = net[absbm_id[0], absbm_id[1], :].reshape(nFeatures, 1)
    return (absbm, absbm_id)

def find_bmu(id_target, target, net):
    """
        Find the best matching unit for a given target vector in the SOM
        Returns: (bmu, bmu_idx) where bmu is the high-dimensional BMU
                 and bmu_idx is the index of this vector in the SOM
    """
    nFeatures = net.shape[2]
    bmu_idx = np.array([0, 0])
    # set the initial minimum distance to a huge number
    #    min_dist = np.iinfo(np.int).max
    #    print("start min dist: ",min_dist)
    min_dist = sys.maxsize
    # calculate the high-dimensional distance between each neuron and the input
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            trial = net[x, y, :].reshape(nFeatures, 1)
            sq_dist = np.sum((trial - target) ** 2)
            if sq_dist < min_dist:
                min_dist = sq_dist
                bmu_idx = np.array([x, y])
    # get vector corresponding to bmu_idx
    bmu = net[bmu_idx[0], bmu_idx[1], :].reshape(nFeatures, 1)
    return (bmu, bmu_idx)

def generateReport(net,lstVars):
    """
    this is the reporter for Self Organizing Maps
        - report occurs at the end of all iterations
        - routine for identifying and reporting membership of Countries in clusters
        - returns the count of Countries in each cluster
    """
    nFeatures = net.shape[2]
    # print("*************  membership of Countries ********************")
    cntGps = np.zeros([nXs,nYs],dtype=np.int)
    print("countries: \n", countries)
    vec=[]
    for iRow in range(nCountrys):
        sov = countries[iRow]
        train = data[:, iRow].reshape(np.array([nFeatures, 1]))
        # absbm, absbm_id = find_absbm3(iRow,train, net)
        absbm, absbm_id = find_bmu(iRow, train, net)
        # print(iRow, np.array(absbm_id)+1, countries[iRow])  ## 
        vec.append([iRow,absbm_id, absbm.T])
        iiX = absbm_id[0]
        iiY = absbm_id[1]
        cntGps[iiX,iiY]+=1
        # print(">>>>****    iRow, iiX, iiY,sov: ", iRow, iiX, iiY, sov)
    for x in range(net.shape[0]):
        for y in range(net.shape[1]):
            train = net[x, y, :].reshape(np.array([nFeatures, 1, 1]))
            iRow+=1
            absbm_id = str([x,y]).replace(',','')
            absbm = train
            vec.append([iRow,absbm_id, absbm.T])
    # print("*************  Features of Countries  ********************")
    # print("Features: \n", lstVars)
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
        - in the final plot, includes the count of the number of Countries in each cluster
    """
    global countries, nFeatures
    patch = []
    axes[0].clear()
    axes[0].annotate(('iter: %d out of %d' % (iIter+1,nCnt)),xy=(0.05,0.95),xycoords='axes fraction')
    nFeatures = net.shape[2]
    nPtMax = 5
    nColorMax = 3
    nColorPts = min(nFeatures,nColorMax)
    orderFace = np.argsort(np.abs(net.std(axis=(0,1))))[::-1][:nFeatures]
    for x in range(1, net.shape[0] + 1):
        for y in range(1, net.shape[1] + 1):
            face = net[x-1,y-1,:]
            # print("face: ", face)
            faceX = []
            for i in range(nColorPts):
                faceX.append(face[orderFace[i]])
            varX = '1st>>'+str(orderFace[0]+1)+": " + \
                  lstVars[orderFace[0]][:9]+' 2nd>>'+str(orderFace[1]+1)+": "+lstVars[orderFace[1]][:9]+ \
                  ' 3rd>>'+str(orderFace[2]+1)+": "+lstVars[orderFace[2]][:9]+' 4th>>'+str(orderFace[3]+1)+ \
                   ": "+lstVars[orderFace[3]][:9]
            axes[0].annotate(varX,xy=(0.05,0.015),xycoords='axes fraction',fontsize=9,fontweight='normal')
            rect = plt.Rectangle((0.05+(x-1.0)/8, 0.014+(y-0.68)/8), 1.0/8.0, 1.0/8.0, facecolor=faceX,edgecolor='gray')
            patch.append(axes[0].add_patch(rect))
            face = [int(1000*face[i])/1000.0 for i in range(nFeatures)]
            strFace = ""
            for i in range(nColx):
                strFace+=(str(orderFace[i]+1)+":  "+str(face[orderFace[i]])+'\n')
            strXYZ = "["+str(x)+","+str(y)+"]"
            if iIter >= nCnt-1:
                cntGp = cntGps[x-1][y-1]
                strXYZ+= ": ("+str(cntGp)+")"
            strFace+=strXYZ
            #-----------------------------------------------------
            colorX = 'orange'
            if faceX[0]>0.7 or faceX[1]>0.6 or faceX[2]>0.75:
                colorX = 'black'
            # axes[0].annotate(strFace,xy=((x-0.78)/(nXs+0.20),(y-0.78)/(nYs+0.20)),xycoords='axes fraction',
            #                  fontsize=7, color=colorX, fontweight='bold')
            axes[0].annotate(strFace,xy=((x-0.45)/(nXs+1),(y-0.55)/(nYs+1)),xycoords='axes fraction',
                         fontsize=7, color=colorX, fontweight='bold')

    if iIter >= nCnt-1 or iIter%5 == 0:
        # print("**************************** plot map for iIter: ", iIter)
        # print("\n*****plotMap")
        plotMap(plt, axes, orderFace)
    return patch


def plotMap(plt, axes, orderFace):
    lstCntClusters = []
    if iIter >= nCnt-1 or iIter%5 == 0:
        axes[1].clear()
        #ax.add_feature(cartopy.feature.LAND)
        axes[1].add_feature(cartopy.feature.OCEAN)
        #ax.add_feature(cartopy.feature.COASTLINE)
        #ax.add_feature(cartopy.feature.BORDERS, linestyle='-', alpha=.5)
        #ax.add_feature(cartopy.feature.LAKES, alpha=0.95)
        #ax.add_feature(cartopy.feature.RIVERS)
        axes[1].set_extent([-150, 60, -25, 60])

        cntGps = np.zeros([nXs,nYs],dtype=np.int)
        lstCountrys = [[[] for i in range(nYs)] for j in range(nXs)]
        vec=[]
        for iRow in range(nCountrys):
            train = data[:, iRow].reshape(np.array([nFeatures, 1]))
            # find its Best Matching Unit
            bmu, bmu_idx = find_bmu(iRow, train, net)
            vec.append([iRow,bmu_idx, bmu.T])
            iiRow = bmu_idx[0]
            iiCol = bmu_idx[1]
            strCountry = countries[iRow]
            # print("strCountry: ", strCountry)
            cntGps[iiRow,iiCol]+=1
            lstCountrys[iiRow][iiCol].append(countries[iRow])
            lstCntClusters.append([iiRow,iiCol])
        if iIter >= nCnt-1:
            for i in range(nYs):
                for j in range(nXs):
                    print("[",j+1,i+1,"]  ",lstCountrys[j][i])
        for x in range(net.shape[0]):
            for y in range(net.shape[1]):
                train = net[x, y, :].reshape(np.array([nFeatures, 1]))
                # set to Best Matching Unit
                # bmu, bmu_idx = find_bmu(train, net, m)
                iRow+=1
                bmu_idx = str([x,y]).replace(',','')
                bmu = train
                vec.append([iRow,bmu_idx, bmu.T])
        unique, counts = np.unique(([str(vec[i][1]) for i in range(len(vec))]), return_counts=True)
        cntList = list(zip(unique,counts))
        cnt = [cntList[i][1] -1 for i in range(len(cntList))]
    Countries = reader.records()
    Countries1 = gpd.read_file('./world/TM_WORLD_BORDERS-0.3.shp')
    
    qTest0 = True
    if qTest0:
        nUsed = 0
        for country in Countries:
            # print("\n ============ country: ", country)
            sov = country.attributes['name']
            lab = country.attributes['adm0_a3']
            # print("sov: ", sov)
            # print("lab: ", lab)
            bounds = country.bounds
            if lab != 'RUS':
                x = ((bounds[0]+bounds[2])/2.0+178.0)/360.0
            else:
                x = ((45.0+bounds[2])/2.0+178.0)/360.0
            y = ((bounds[1]+bounds[3])/2.0+59)/145.0
            valXY = (x,y)
            if sov in countries:
                nUsed+=1
                ind1 = countries.index(sov)
                # print("ind1: ",ind1)
                ind2 = lstCntClusters[ind1]
                # print("ind2: ",ind2)
                color = (net[ind2[0],ind2[1],0],net[ind2[0],ind2[1],1],net[ind2[0],ind2[1],2])
                color = (net[ind2[0],ind2[1],orderFace[0]],net[ind2[0],ind2[1],orderFace[1]],net[ind2[0],ind2[1],orderFace[2]])
                # print(">>>>****    sov, ind1, ind2, color, nUsed:  ",sov, ind1, ind2, color, nUsed)
                axes[1].add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor = color, label=lab)
                colorX = 'orange'
                axes[1].annotate(lab,valXY,xycoords='axes fraction',fontsize=8,color=colorX,fontweight='bold')
            else:
                axes[1].add_geometries(country.geometry, ccrs.PlateCarree(),
                          facecolor=(0.8, 0.8, 0.8), label=lab)
                colorX = 'grey'
                axes[1].annotate(lab,valXY,xycoords='axes fraction',fontsize=8,color=colorX,fontweight='bold')


    qTest1 = False
    if qTest1:
        nUsed = 0
        sovs = Countries1.NAME
        labs = Countries1.ISO3
        geos = Countries1.geometry
        nCountries = len(sovs)
        for iCountry in range(nCountries):
            sov = sovs[iCountry]
            lab = labs[iCountry]
            geo = geos[iCountry]
            bounds = geo.bounds
            print("sov: ", sov)
            print("lab: ", lab)
            print("bounds: ", bounds)
            if lab != 'RUS':
                x = ((bounds[0]+bounds[2])/2.0+178.0)/360.0
            else:
                x = ((45.0+bounds[2])/2.0+178.0)/360.0
            y = ((bounds[1]+bounds[3])/2.0+59)/145.0
            valXY = (x,y)
            if sov in countries:
                nUsed+=1
                ind1 = countries.index(sov)
                # print("ind1: ",ind1)
                ind2 = lstCntClusters[ind1]
                # print("ind2: ",ind2)
                color = (net[ind2[0],ind2[1],0],net[ind2[0],ind2[1],1],net[ind2[0],ind2[1],2])
                color = (net[ind2[0],ind2[1],orderFace[0]],net[ind2[0],ind2[1],orderFace[1]],net[ind2[0],ind2[1],orderFace[2]])
                # print(">>>>****    sov, ind1, ind2, color, nUsed:  ",sov, ind1, ind2, color, nUsed)
                axes[1].add_geometries(geo, ccrs.PlateCarree(), facecolor = color, label=lab)
                colorX = 'orange'
                axes[1].annotate(lab,valXY,xycoords='axes fraction',fontsize=8,color=colorX,fontweight='bold')
            else:
                axes[1].add_geometries(geo, ccrs.PlateCarree(),
                          facecolor=(0.8, 0.8, 0.8), label=lab)
                colorX = 'grey'
                axes[1].annotate(lab,valXY,xycoords='axes fraction',fontsize=8,color=colorX,fontweight='bold')


    Countries.close()
    plt.gca().set_yticks([-60, -30, 0, 30, 60], crs=ccrs.PlateCarree())
    plt.gca().set_xticks(np.arange(-180,240,60), crs=ccrs.PlateCarree())
    plt.gca().gridlines()
    return

def decay_radius(initial_radius, i, time_constant):
    return initial_radius * np.exp(-i / time_constant)

def decay_learning_rate(initial_learning_rate, i, n_iterations):
    return initial_learning_rate * np.exp(-i / n_iterations)

def calculate_influence(distance, radius):
    return np.exp(-distance / (2* (radius**2)))

def getRawData():
    """
    this routine pulls in the raw data
    """
    import csv
    df = pd.read_csv('./Data/factbook.csv', sep=';')
    lstVars = list(df.columns[1:45])
    vars1 = ['Area(sq km)','Birth rate(births/1000 population)','Death rate(deaths/1000 population)',
             'Infant mortality rate(deaths/1000 live births)','Life expectancy at birth(years)','Population']
    vars2 = ['Electricity - consumption(kWh)','Electricity - production(kWh)','Exports','Highways(km)','Imports',
                'Internet users','Oil - consumption(bbl/day)','Oil - production(bbl/day)',
                'Telephones - main lines in use','Telephones - mobile cellular']
    vars3 = ['Debt - external','GDP','GDP - per capita','GDP - real growth rate(%)',
                'Inflation rate (consumer prices)(%)']
    lstVars = vars1+vars2+vars3
    subset = df[lstVars].drop(0)
    xCnt = list(len(subset.loc[pd.notnull(df[lstVars[i]])]) for i in range(len(lstVars)))
    nCutOff = 160
    lstVars = [lstVars[i] for i in range(len(lstVars)) if xCnt[i] > nCutOff]
    subset = subset[lstVars]
    print(lstVars)
    for i in range(len(lstVars)):
        print(i,lstVars[i])
        subset = subset.loc[pd.notnull(df[lstVars[i]])]
    countries = list(df['Country'][subset.index][1:])
    print("countries: ",countries)
    subset = subset.astype(float)
    # subset = subset.convert_objects(convert_numeric=True)
    groups = df[['Country']]
    
    tuples = [tuple(x) for x in subset.values[1:,:]]
    nRows = len(tuples)
    raw_data = np.transpose(tuples)
    print("raw_data.shape: ", raw_data.shape)
    print("raw data: ", raw_data)
    return (lstVars, countries, raw_data)

def animate(iIter):
    """
    this is the heart of Self Organizing Maps
        - picks a random oil field 
        - finds the vector of the 3 by 3 by 4 net which is closest
        - having found the closest look around that vector and adjust those around it to reflect
            - radius:  how far do we search
            - learn:  how much are we able to learn
            - train:  transformed real data that trains the net
            - trial:  net vector that keeps iterating closer and closer to strong clusters
            - influence:  the closer other vectors are, the more they are influenced
            - learn:  the more often the cycles occur, the less there is to learn and learning rate drops
            - diff = trial - train:  the bigger the diff between the train and the trial data the bigger the adjustment required
            - net effect is summarized as:  (trial + (learn * influence * (train - trial)))
        - any decreasing functions can be used for influence and learn (so long as they decrease over iterations)
    """
    global net, axes, nCnt, nRowsPerIter, qInit
    global nYs, nXs, nZs
    nRowsThisIter = nRowsPerIter
    qInit=False
    if qInit:
        nRowsThisIter = 1
    qInit = False
    if iIter >= 9999999:
        import sys
        sys.exit(0)
    nFeatures = net.shape[2]
    nColx = nFeatures    #  how many figures to annotate
    if nColx > 5:
        nColx = 5
    initRadiusM1 = init_radius - 1
    #  print("nRowsThisIter: ", nRowsThisIter)
    for iX in range(nRowsThisIter):
        # identify a random oil field
        intRnd = np.random.randint(0, nCountrys)
        train = data[:, intRnd].reshape(np.array([nFeatures, 1]))
        # find closest location to properties of that oil field
        # absbm, absbm_id = find_absbm3(intRnd, train, net)
        absbm, absbm_id = find_bmu(intRnd, train, net)
        # print("iX, bmu, bmu_idx:", iX, absbm.T, absbm_id)  ##########################################
        # figure out how far into the process we have gone
        iStep = nRowsThisIter*iIter+iX
        # adjust the radius for how far we have advanced
        # calculate radius of search
        radix = decay_radius(init_radius, nRowsPerIter*iIter+iX, time_constant)
        ### radix = 1 + initRadiusM1 * (n_iterations - iStep)/n_iterations   ### small difference
        # find square of radius for re-use
        radi_radi = radix * radix
        # adjust learning rate based on how far we have advanced
        learn = decay_learning_rate(ipt1_learning_rate, nRowsPerIter*iIter+iX, n_iterations)
        ### learn = ipt2_learning_rate + (ipt1_learning_rate - ipt2_learning_rate)*(1.0 - iStep / n_iterations)   ### medium difference
        # run thru 3 dimensions
        for x in range(net.shape[0]):
            for y in range(net.shape[1]):
                wt_dist = np.sum((np.array([x, y]) - absbm_id) ** 2)
                # wt_dist = np.sum(abs((np.array([x, y, z]) - absbm_id)))
                # if wt_dist <= radix:
                if wt_dist <= radi_radi:
                    trial = net[x, y, :].reshape(nFeatures, 1)
                    # influence depends on how far away another vector is
                    ### influence = 1.0 - 0.5 * (wt_dist/(radi_radi))    ### small difference
                    influence = calculate_influence(wt_dist, radix)
                    net[x, y, :] = (trial + (learn * influence * (train - trial))).reshape(1, nFeatures)
    axes[0].clear()
    axes[0].annotate(('iter: %d out of %d     top %d out of %d Features displayed' % (nRowsPerIter*(iIter+1),nRowsPerIter*nCnt,nColx,nFeatures)),
                     xy=(0.05,0.985),xycoords='axes fraction',fontsize=8,fontweight='normal')
    cntGps = []
    if iIter >= nCnt-1:
        cntGps = generateReport(net,lstVars)
    # print("\n **************before patch iIter,cntGps,nColx: ",iIter,cntGps,nColx)
    patch = generatePlot(iIter,net,cntGps,nColx)

    if iIter >= nCnt-1:
        print("******* net: ")
        print(net)
        end_time = int(round(time.time() * 1000))
        diff_time = (end_time - start_time)/1000.0
        print (diff_time)
    return patch


np.random.seed(123456)

axes=[None]*2

# import matplotlib.path as path

#   %matplotlib inline

start_time = int(round(time.time() * 1000))
np.set_printoptions(suppress=True)
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
               
iIter = 0
# nRows = 1000
nRowsPerIter = 100
n_iterations = 12000
# n_iterations = 3800    ################# set temporary count...  ##################


lstVars,countries,raw_data = getRawData()

nDims = 2
nYs = 7
nXs = 7

net_dims = np.array([nYs, nXs])
nCnt = int(n_iterations/nRowsPerIter)
# nCnt = 1
ipt1_learning_rate = 0.03
ipt2_learning_rate = 0.01
weight_range = [0, 1]
a = weight_range[0]
b = weight_range[1]

# establish size variables based on data
nFeatures = raw_data.shape[0]
nCountrys = raw_data.shape[1]
print('>>>>>>>>>>>>> nFeatures,nCountrys: ',nFeatures,nCountrys)
# initial neighbourhood radius
init_radius = max(net_dims[0], net_dims[1]) / 2
# radius decay parameter
time_constant = n_iterations / np.log(init_radius)
# we want to keep a copy of the raw data for later
data = raw_data
# check if data needs to be normalised
print("********************  data: ")
print(data)
# normalise along each column
col_maxes = raw_data.max(axis=1)
col_mines = raw_data.min(axis=1)
data = (raw_data - col_mines[:, np.newaxis]) / ((col_maxes-col_mines)[:, np.newaxis])
net = (b-a) * np.random.random((net_dims[0], net_dims[1], nFeatures)) + a
print("***    net    ***")
print(net)
fig_size = plt.rcParams["figure.figsize"]
print("Current size of plot:", fig_size)
fig_size[0]=18
fig_size[1]=8
plt.rcParams["figure.figsize"] = fig_size
fig = plt.figure()
gs = gridspec.GridSpec(1,2,width_ratios = [3,5])
axes[0] = fig.add_subplot(gs[0], aspect='equal')

axes[0].set_xlim((0, net.shape[0]+1))
axes[0].set_ylim((0, net.shape[1]+1))

axes[1] = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
axes[1].add_feature(cartopy.feature.OCEAN)
axes[1].set_extent([-150, 60, -25, 60])

# shpfilename = shpreader.natural_earth(resolution='110m',category='cultural',name='admin_0_countries')
shpfilename = './cartopy/shapefiles/natural_earth/cultural/110m_admin_0_countries.shp'
print("shpfilename: ", shpfilename)
reader = shpreader.Reader(shpfilename)

plt.tight_layout()
qInit = True

ani = animation.FuncAnimation(fig, animate, nCnt, repeat=False, blit=False)
plt.show()
