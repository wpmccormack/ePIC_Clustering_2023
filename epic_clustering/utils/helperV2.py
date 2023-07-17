import numpy as np
from truthCluster import truthCluster
from superCluster import superCluster
from singleLayerCluster import singleLayerCluster
from multiDepthCluster import multiDepthCluster



def makeTruthClusters(event):
    truthClusters = []
    #This is more verbose than I would like, but the variables of interest (ID and Frac) have different, hard-coded names
    tmpTCs = []
    for i in range(max(event.tower_LFHCAL_trueID1)+1):
        hits1 = np.argwhere(event.tower_LFHCAL_trueID1==i)
        if(len(hits1)>1 or len(hits1)==0):
            hits1 = np.squeeze(hits1)
        else:
            hits1 = hits1[0]
        hits2 = np.argwhere(event.tower_LFHCAL_trueID2==i)
        if(len(hits2)>1 or len(hits2)==0):
            hits2 = np.squeeze(hits2)
        else:
            hits2 = hits2[0]
        hits3 = np.argwhere(event.tower_LFHCAL_trueID3==i)
        if(len(hits3)>1 or len(hits3)==0):
            hits3 = np.squeeze(hits3)
        else:
            hits3 = hits3[0]
        hits4 = np.argwhere(event.tower_LFHCAL_trueID4==i)
        if(len(hits4)>1 or len(hits4)==0):
            hits4 = np.squeeze(hits4)
        else:
            hits4 = hits4[0]

        e1 = event.tower_LFHCAL_E[hits1]
        e2 = event.tower_LFHCAL_E[hits2]
        e3 = event.tower_LFHCAL_E[hits3]
        e4 = event.tower_LFHCAL_E[hits4]
        f1 = event.tower_LFHCAL_trueEfrac1[hits1]
        f2 = event.tower_LFHCAL_trueEfrac2[hits2]
        f3 = event.tower_LFHCAL_trueEfrac3[hits3]
        f4 = event.tower_LFHCAL_trueEfrac4[hits4]

        tmpE = sum(np.multiply(e1,f1)) + sum(np.multiply(e2,f2)) + sum(np.multiply(e3,f3)) + sum(np.multiply(e4,f4))
        if(tmpE > 0):
            tmpX = sum(np.multiply(np.multiply(e1,f1), event.tower_LFHCAL_posx[hits1]))
            tmpX += sum(np.multiply(np.multiply(e2,f2), event.tower_LFHCAL_posx[hits2]))
            tmpX += sum(np.multiply(np.multiply(e3,f3), event.tower_LFHCAL_posx[hits3]))
            tmpX += sum(np.multiply(np.multiply(e4,f4), event.tower_LFHCAL_posx[hits4]))
            tmpX = tmpX/tmpE
            tmpY = sum(np.multiply(np.multiply(e1,f1), event.tower_LFHCAL_posy[hits1]))
            tmpY += sum(np.multiply(np.multiply(e2,f2), event.tower_LFHCAL_posy[hits2]))
            tmpY += sum(np.multiply(np.multiply(e3,f3), event.tower_LFHCAL_posy[hits3]))
            tmpY += sum(np.multiply(np.multiply(e4,f4), event.tower_LFHCAL_posy[hits4]))
            tmpY = tmpY/tmpE
            tmpZ = sum(np.multiply(np.multiply(e1,f1), event.tower_LFHCAL_posz[hits1]))
            tmpZ += sum(np.multiply(np.multiply(e2,f2), event.tower_LFHCAL_posz[hits2]))
            tmpZ += sum(np.multiply(np.multiply(e3,f3), event.tower_LFHCAL_posz[hits3]))
            tmpZ += sum(np.multiply(np.multiply(e4,f4), event.tower_LFHCAL_posz[hits4]))
            tmpZ = tmpZ/tmpE
            tmpTCs.append(truthCluster(i, tmpE, tmpX, tmpY, tmpZ, list(hits1)+list(hits2)+list(hits3)+list(hits4), list(f1)+list(f2)+list(f3)+list(f4)))

    tmpTCs.sort(key=lambda c: -c.energy)

    #I combine truth particles if they're too close
    #Clusters are stochastic enough that discovering a small "sub-cluster" within a larger cluster is very hard from a reconstruction point of view (though perhaps not impossible)
    keptTCs = [0]
    for tc2 in range(1,len(tmpTCs)):
        absorbed = False
        for tc1 in range(tc2):
            tcDist = calcDist(tmpTCs[tc1], tmpTCs[tc2])
            if(tmpTCs[tc2].energy < tmpTCs[tc1].energy*np.exp(-0.5 * np.power(tcDist/10.,2))):
                tmpTCs[tc1].combineTwoTruthClusters(tmpTCs[tc2])
                absorbed = True
                break
        if(not absorbed and tmpTCs[tc2].energy > 0.5):
            keptTCs.append(tc2)

    for k in keptTCs:
        truthClusters.append(tmpTCs[k])

    truthClusters.sort(key=lambda c: -c.energy)
    return truthClusters


def runClusterizer(event):
    #Hits are sorted.
    #expandCluster is a recursive algorithm that will mask hits as they're used in superClusters
    listOfSCs = []
    for a in event.args:
        if(not event.mask[a]):
            continue
        event.mask[a] = False
        tmpSC = superCluster(event, event.tower_LFHCAL_iz[a])
        expandCluster(event, a, tmpSC)
        tmpSC.calculateCluster()
        listOfSCs.append(tmpSC)

    listOfSCs.sort(key=lambda sc: -sc.energy)
    return listOfSCs


def expandCluster(event, argInQuestion, superCluster):

    superCluster.appendHit(argInQuestion)
    currX = event.tower_LFHCAL_ix[argInQuestion]
    currY = event.tower_LFHCAL_iy[argInQuestion]
    currZ = event.tower_LFHCAL_iz[argInQuestion]

    #Here I scan a grid of adjacent indices from two indices to the left to two to the right, and from 2 down to two up
    #This scan is probably too large, but during development, the indexing isn't great.
    #If you're counting, this scan checks 24 other indices (5x2 grid on the left, 5x2 grid on the right, and 2x1 up and 2x1 down)
    #In CMS for example, the default is only 4 other indices: one left, one right, one up, and one down, but I trust the indexing there
    for i in range(-2, 3):
        for j in range(-2, 3):
            if(not (i==0 and j==0) and currX+i >=0 and currY+j >=0):
                testTuple = (currX+i, currY+j, currZ)
                if(testTuple in event.indexDict_Layered and event.mask[event.indexDict_Layered[testTuple]]):
                    neighborArg = event.indexDict_Layered[testTuple]
                    event.mask[neighborArg] = False
                    #superCluster.event = event
                    expandCluster(event, neighborArg, superCluster)


def combineSuperClustersInLayer(listOfSCs):
    combinedListOfSCs = []
    for i in range(len(listOfSCs)):
        new = True
        for s in range(len(combinedListOfSCs)):
            if(not listOfSCs[i].iz == combinedListOfSCs[s].iz):
                continue
            if(calcDist(listOfSCs[i],combinedListOfSCs[s]) < 20.):
                combinedListOfSCs[s].appendListOfHits(listOfSCs[i].hitIndices)
                combinedListOfSCs[s].calculateCluster()
                new = False
                break
        if(new):
            combinedListOfSCs.append(listOfSCs[i])

    combinedListOfSCs.sort(key=lambda sc: -sc.energy)

    return combinedListOfSCs


def makeAllClusters(event, combinedListOfSCs):
    #Here we're making smaller, "regular" clusters that only have one distinct seed in them.
    #Regular clusters can share hits with other regular clusters that both come from the same superCluster, but with energy fractions determined by the distance from hit to cluster center
    #I should probably add in some cluster "pruning" to remove hits that are far from the centroid or contribute only a small fraction of their energy to the cluster

    allClusters = []
    
    for c in combinedListOfSCs:
        if(c.energy < 0.1):
            break
        allClusters += splitSuperCluster(event, c)

    allClusters.sort(key=lambda sc: -sc.energy)
    return allClusters


def splitSuperCluster(event, SC):
    arrayOfClusters = []
    seedsInSC = np.intersect1d(event.seeds, SC.hitIndices)
    numSeedsInSC = len(seedsInSC)

    if(numSeedsInSC <= 1):
        tmpCluster = singleLayerCluster(event, SC.iz, SC.hitIndices)
        tmpCluster.setHitFrac([1]*len(SC.hitIndices))
        tmpCluster.calculateCluster()
        arrayOfClusters.append(tmpCluster)
    else:
        #Split the superCluster
        hitsInSC = SC.hitIndices
        scHitMask = [True]*(max(hitsInSC)+1)
        for s in seedsInSC:
            #Initialize split clusters using the seeds in the superCluster
            tmpCluster = singleLayerCluster(event, SC.iz, [s])
            tmpCluster.appendHitFrac(1)
            tmpCluster.calculateCluster()
            arrayOfClusters.append(tmpCluster)
            scHitMask[s] = False
        for h in hitsInSC:
            if(not scHitMask[h]):
                continue
            #Energy fraction assigment based on distance.  Using shower sigma of 10 now
            sumEnergyContrib = 0.
            for c in arrayOfClusters:
                sumEnergyContrib += c.energy*np.exp(-0.5 * np.power(c.dist(h)/10.,2))
            for c in arrayOfClusters:
                c.appendHit(h)
                c.appendHitFrac(c.energy*np.exp(-0.5 * np.power(c.dist(h)/10.,2))/sumEnergyContrib)
                c.calculateCluster()

    return arrayOfClusters


def makeLayeredClusters(allClusters):
    layeredSLCs = []
    izs = [c.iz for c in allClusters]
    for i in range(7):
        if(len(np.argwhere(np.asarray(izs)==i)) > 1):
            inlayer = list(np.squeeze(np.argwhere(np.asarray(izs)==i)))
            layeredSLCs.append(list(np.asarray(allClusters)[inlayer]))
        elif(len(np.argwhere(np.asarray(izs)==i)) == 1):
            inlayer = np.squeeze(np.argwhere(np.asarray(izs)==i))
            layeredSLCs.append([np.asarray(allClusters)[inlayer]])
        else:
            layeredSLCs.append([])

        layeredSLCs[i].sort(key=lambda c: -c.energy)

    return layeredSLCs


def makeMultiDepthClusters(event, layeredSLCs):
    multiDepthClusters = []
    for i in range(7):
        for c in layeredSLCs[i]:
            used = False
            for m in multiDepthClusters:
                if(used):
                    break
                if(not c.iz in m.layerSet):
                    for l in range(c.iz):
                        if(used):
                            break
                        if( (c.iz - (l+1)) in m.layerSet):
                            #layer-to-layer displacement increases as you move outward
                            #E.g. for particles out at e.g. posx = 200, I noticed that the energy might be centered at 190 in layer 0, 200 in layer 1, 210 in layer 2 and so on
                            #In the center of the detectory, the energy does not have much layer-to-layer displacement
                            extrapolatedX = c.posx - (l+1) * (c.posx/20.)
                            extrapolatedY = c.posy - (l+1) * (c.posy/20.)
                            if(m.distClustLayerExtrapolate(extrapolatedX, extrapolatedY, (c.iz - (l+1))) < 15.):
                                m.addSingleLayerCluster(c)
                                m.calculateCluster()
                                used = True
            if(not used):
                tmpMDC = multiDepthCluster(event, c)
                tmpMDC.calculateCluster()
                multiDepthClusters.append(tmpMDC)

    multiDepthClusters.sort(key=lambda c: -c.energy)

    return multiDepthClusters


def doClusterMatching(truthClusters, multiDepthClusters):
    matchedClusters = []
    #initialization, just in case
    for tc in truthClusters:
        tc.matched = False
    for rc in multiDepthClusters:
        rc.truthMatch = -1 #dummy value, since truthMatch points to the truth particle's ID value
    for tc in range(len(truthClusters)):
        for rc in range(len(multiDepthClusters)):
            if(calcDist(truthClusters[tc], multiDepthClusters[rc]) < 15
               and not(truthClusters[tc].matched)
               and (multiDepthClusters[rc].truthMatch < 0)):
                multiDepthClusters[rc].truthMatch = truthClusters[tc].label
                truthClusters[tc].matched = True
                matchedClusters.append((tc,rc))
                break
        if(not(truthClusters[tc].matched)):
            matchedClusters.append((tc,None))
    for rc in range(len(multiDepthClusters)):
        if(multiDepthClusters[rc].truthMatch<0):
            matchedClusters.append((None,rc))

    return matchedClusters
            
def findMatchedDiffs(matchedClusters, truthClusters, multiDepthClusters):
    diffs = []
    for mc in matchedClusters:
        if(not mc[0] is None and not mc[1] is None):
            diffs.append((truthClusters[mc[0]].energy - multiDepthClusters[mc[1]].energy)/truthClusters[mc[0]].energy)

    return diffs



                    
def calcDist(sc1, sc2):
    return np.sqrt(np.power(sc1.posx-sc2.posx,2)+np.power(sc1.posy-sc2.posy,2))

def calcDistRaw(x1, y1, x2, y2):
    return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
