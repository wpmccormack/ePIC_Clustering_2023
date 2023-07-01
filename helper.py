import numpy as np
import ROOT
from truthCluster import truthCluster
from superCluster import superCluster
from singleLayerCluster import singleLayerCluster
from multiDepthCluster import multiDepthCluster

def expandCluster(event, rhc, argInQuestion, superCluster):
    if(argInQuestion in event.args):
        event.args = np.delete(event.args, np.argwhere(event.args==argInQuestion))
    if(argInQuestion in superCluster.hitIndices):
        return
    superCluster.appendHit(argInQuestion)
        
    #for i,j in [(-1,0), (0,-1), (1,0), (0,1)]: #Explore neighboring hits (but not diagonally adjacent hits)
    #for i,j in [(-1,0), (0,-1), (1,0), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]: #explore 8 adjacent hits
    for i in range(-2, 3):
        for j in range(-2, 3):
            if(not (i==0 and j==0)):
                if((event.tower_LFHCAL_ix[argInQuestion]+i, event.tower_LFHCAL_iz[argInQuestion]) in rhc.ixdict_layered and (event.tower_LFHCAL_iy[argInQuestion]+j, event.tower_LFHCAL_iz[argInQuestion]) in rhc.iydict_layered):
                    neighborhood = np.intersect1d(rhc.ixdict_layered[event.tower_LFHCAL_ix[argInQuestion]+i, event.tower_LFHCAL_iz[argInQuestion]], rhc.iydict_layered[event.tower_LFHCAL_iy[argInQuestion]+j, event.tower_LFHCAL_iz[argInQuestion]])
                    for n in neighborhood:
                        if(n in event.args):
                            event.args = np.delete(event.args, np.argwhere(event.args==n))
                            expandCluster(event, rhc, n, superCluster)


                            
def calcDist(sc1, sc2):
    return np.sqrt(np.power(sc1.posx-sc2.posx,2)+np.power(sc1.posy-sc2.posy,2))

def calcDistRaw(x1, y1, x2, y2):
    return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))



def runClusterizer(event, rhc):

    listOfSCs = []

    while(len(event.args) > 0):
        tmpSC = superCluster(event, rhc)
        expandCluster(event, rhc, event.args[0], tmpSC)
        tmpSC.event = event
        tmpSC.calculateCluster()
        listOfSCs.append(tmpSC)
        
    listOfSCs.sort(key=lambda sc: -sc.energy)

    return listOfSCs



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


def combineSuperClusters(listOfSCs):
    combinedListOfSCs = []
    for i in range(len(listOfSCs)):
        new = True
        for s in range(len(combinedListOfSCs)):
            if(calcDist(listOfSCs[i],combinedListOfSCs[s]) < 20.):
                combinedListOfSCs[s].appendListOfHits(listOfSCs[i].hitIndices)
                combinedListOfSCs[s].calculateCluster()
                new = False
                break
        if(new):
            combinedListOfSCs.append(listOfSCs[i])

    combinedListOfSCs.sort(key=lambda sc: -sc.energy)
            
    return combinedListOfSCs


"""
def clusterMergeCheck(cluster):
    testy = ROOT.TH1D("testY", "testY", 80, -200, 200)
    testx = ROOT.TH1D("testX", "testX", 80, -200, 200)
    for h in cluster.hitIndices:
        testy.Fill(cluster.event.tower_LFHCAL_posy[h])
        testx.Fill(cluster.event.tower_LFHCAL_posx[h])
    
    satisfied = False
    extratests = 0
    yFuncs = []
    xFuncs = []
    p = 1
    bestP = 1
    currprobX = -1.
    currprobY = -1.

    while((not satisfied or extratests < 3) and p<10):
        #Check the number of gaussians that seem best suited to fit the cluster
        #Easiest in 1D, though this could be done with 2D histograms
        #Currently will allow up to 9 peaks in the cluster
        #Checks a few extra peaks after one is accepted to make sure that there isn't a slightly better configuration.  This might need to be tweaked.
        
        yFuncs.append(multiGaus(p, "testFy_"+str(p), cluster.posy))
        testy.Fit("testFy_"+str(p), "q")
        fitresultY = testy.GetFunction("testFy_"+str(p))
        
        xFuncs.append(multiGaus(p, "testFx_"+str(p), cluster.posx))
        testx.Fit("testFx_"+str(p), "q")
        fitresultX = testx.GetFunction("testFx_"+str(p))
        
        if(not(satisfied) and (fitresultY.GetProb() > 0.1 and fitresultX.GetProb() > 0.1)):
            satisfied = True
            currprobY = fitresultY.GetProb()
            currprobX = fitresultX.GetProb()
            bestP = p
        if(satisfied and (fitresultY.GetProb()/currprobY > 2. or fitresultX.GetProb()/currprobX > 2.)):
            extratests = 0
            bestP = p
            currprobY = fitresultY.GetProb()
            currprobX = fitresultX.GetProb()
        if(satisfied and not (fitresultY.GetProb()/currprobY > 2. or fitresultX.GetProb()/currprobX > 2.)):
            extratests += 1
        p += 1

    return bestP, xFuncs[bestP-1], yFuncs[bestP-1]
"""

def clusterMergeCheck(cluster):
    testy = ROOT.TH1D("testY", "testY", 80, -200, 200)
    testx = ROOT.TH1D("testX", "testX", 80, -200, 200)
    for h in cluster.hitIndices:
        testy.Fill(cluster.event.tower_LFHCAL_posy[h])
        testx.Fill(cluster.event.tower_LFHCAL_posx[h])
    
    yFuncs = []
    xFuncs = []
    bestPY = 1
    bestPX = 1
    currprobX = -1.
    currprobY = -1.

    p = 1
    satisfied = False
    extratests = 0
    bestChi2 = 0
    bestNDF = 0
    while((not satisfied or extratests < 2) and p<10):
        #Check the number of gaussians that seem best suited to fit the cluster
        #Easiest in 1D, though this could be done with 2D histograms
        #Currently will allow up to 9 peaks in the cluster
        #Checks a few extra peaks after one is accepted to make sure that there isn't a slightly better configuration.  This might need to be tweaked.
        #Should probably implement a real f-test for deciding next step
        
        yFuncs.append(multiGaus(p, "testFy_"+str(p), cluster.posy))
        testy.Fit("testFy_"+str(p), "q")
        fitresultY = testy.GetFunction("testFy_"+str(p))
        
        if(not(satisfied) and fitresultY.GetProb() > 0.05):
            satisfied = True
            currprobY = fitresultY.GetProb()
            bestPY = p
            bestChi2 = fitresultY.GetChisquare()
            bestNDF = fitresultY.GetNDF()
        #if(satisfied and fitresultY.GetProb()/currprobY > 2.):
        elif(satisfied and Ftest(bestChi2, bestNDF, fitresultY.GetChisquare(), fitresultY.GetNDF()) < 0.05):
            extratests = 0
            bestPY = p
            currprobY = fitresultY.GetProb()
            bestChi2 = fitresultY.GetChisquare()
            bestNDF = fitresultY.GetNDF()
        #if(satisfied and not fitresultY.GetProb()/currprobY > 2.):
        elif(satisfied and not Ftest(bestChi2, bestNDF, fitresultY.GetChisquare(), fitresultY.GetNDF()) < 0.05):
            extratests += 1
        p += 1

        
    p = 1
    satisfied = False
    extratests = 0
    bestChi2 = 0
    bestNDF = 0
    while((not satisfied or extratests < 2) and p<10):
        #Check the number of gaussians that seem best suited to fit the cluster
        #Easiest in 1D, though this could be done with 2D histograms (ROOT 2D fitting is not easy)
        #Currently will allow up to 9 peaks in the cluster
        #Checks a few extra peaks after one is accepted to make sure that there isn't a slightly better configuration.  This might need to be tweaked.
        #Should probably implement a real f-test for deciding next step
        
        xFuncs.append(multiGaus(p, "testFx_"+str(p), cluster.posx))
        testx.Fit("testFx_"+str(p), "q")
        fitresultX = testx.GetFunction("testFx_"+str(p))
        
        if(not(satisfied) and fitresultX.GetProb() > 0.05):
            satisfied = True
            currprobX = fitresultX.GetProb()
            bestPX = p
            bestChi2 = fitresultX.GetChisquare()
            bestNDF = fitresultX.GetNDF()
        #if(satisfied and fitresultX.GetProb()/currprobX > 2.):
        elif(satisfied and Ftest(bestChi2, bestNDF, fitresultX.GetChisquare(), fitresultX.GetNDF()) < 0.05):
            extratests = 0
            bestPX = p
            currprobX = fitresultX.GetProb()
            bestChi2 = fitresultX.GetChisquare()
            bestNDF = fitresultX.GetNDF()
        #if(satisfied and not fitresultX.GetProb()/currprobX > 2.):
        elif(satisfied and not Ftest(bestChi2, bestNDF, fitresultX.GetChisquare(), fitresultX.GetNDF()) < 0.05):
            extratests += 1
        p += 1

    return bestPX, bestPY, xFuncs[bestPX-1], yFuncs[bestPY-1]



def multiGaus(peaks, name, pos):
    funcstr = "[%i]*exp(-0.5*((x-[%i])/[%i])^2)" % tuple(range(0,3))
    for p in range(1,int(peaks)):
        funcstr += " + [%i]*exp(-0.5*((x-[%i])/[%i])^2)" % tuple(range(3*p,3*p+3))
    returnFunc = ROOT.TF1(name, funcstr, -150, 150)
    for i in range(peaks*3):
        if(i%3 == 0):
            returnFunc.SetParameter(i, 1)
        if(i%3 == 1):
            returnFunc.SetParameter(i, pos - 5*peaks + (i/3)*10)
            returnFunc.SetParLimits(i, -150, 150)
        if(i%3 == 2):
            returnFunc.SetParameter(i, 10)
    return returnFunc


def Ftest(chi2_base, nDof_base, chi2_new, nDof_new):
    F_num =   max((chi2_base - chi2_new), 0)/(abs(nDof_new - nDof_base))
    F_denom = chi2_new/nDof_new
    F = F_num / F_denom
    
    prob = 1. - ROOT.TMath.FDistI(F, abs(nDof_new - nDof_base), nDof_new)
    return prob


def splitSuperCluster(event, rhc, superCluster):
    arrayOfClusters = []
    seedsInSC = np.intersect1d(event.seeds, superCluster.hitIndices)
    numSeedsInSC = len(seedsInSC)
    
    if(numSeedsInSC <= 1):
        tmpCluster = singleLayerCluster(event, rhc)
        tmpCluster.appendListOfHits(superCluster.hitIndices)
        tmpCluster.setHitFrac([1]*len(superCluster.hitIndices))
        tmpCluster.calculateCluster()
        arrayOfClusters.append(tmpCluster)
    else:
        #Split the superCluster
        hitsInSC = superCluster.hitIndices
        for s in seedsInSC:
            #Initialize split clusters using the seeds in the superCluster
            tmpCluster = singleLayerCluster(event, rhc)
            tmpCluster.appendHit(s)
            tmpCluster.appendHitFrac(1)
            tmpCluster.calculateCluster()
            arrayOfClusters.append(tmpCluster)
            hitsInSC = np.delete(hitsInSC, np.argwhere(hitsInSC==s))
        for h in hitsInSC:
            #Energy fraction assigment based on distance.  Using shower sigma of 10 now
            sumEnergyContrib = 0.
            for c in arrayOfClusters:
                sumEnergyContrib += c.energy*np.exp(-0.5 * np.power(c.dist(h)/10.,2))
            for c in arrayOfClusters:
                c.appendHit(h)
                c.appendHitFrac(c.energy*np.exp(-0.5 * np.power(c.dist(h)/10.,2))/sumEnergyContrib)
                c.calculateCluster()
    
    return arrayOfClusters


def allClusters(event, rhc, combinedListOfSCs):
    allClusters = []
    for c in combinedListOfSCs:
        if(c.energy < 0.1):
            break
        allClusters += splitSuperCluster(event, rhc, c)

    return allClusters

def layeredClusters(allClusters):
    izs = [c.iz for c in allClusters]
    layeredSLCs = []
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


def multiDepthClusters(layeredClusters):
    multiDepthClusters = []
    for i in range(7):
        for c in layeredClusters[i]:
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
                            extrapolatedX = c.posx - (l+1) * (c.posx/20.)
                            extrapolatedY = c.posy - (l+1) * (c.posy/20.)
                            if(m.distClustLayerExtrapolate(extrapolatedX, extrapolatedY, (c.iz - (l+1))) < 15.):
                                m.addSingleLayerCluster(c)
                                m.calculateCluster()
                                used = True
            if(not used):
                multiDepthClusters.append(multiDepthCluster(c))

    multiDepthClusters.sort(key=lambda c: -c.energy)
                
    return multiDepthClusters


def makeTruthClusters(event):
    truthClusters = []
    for i in range(max(event.tower_LFHCAL_trueID1)+1):
        
        hitIndices = np.squeeze(np.argwhere(event.tower_LFHCAL_trueID1==i))
        if(sum(event.tower_LFHCAL_E[hitIndices]) > 0):
            tmpcl = truthCluster(i, event)
            truthClusters.append(tmpcl)
    truthClusters.sort(key=lambda c: -c.energy)

    return truthClusters



def doClusterMatching(truthClusters, multiDepthClusters):
    matchedClusters = []
    for tc in truthClusters:
        tc.matched = False
    for rc in multiDepthClusters:
        rc.truthMatch = -1
    for tc in range(len(truthClusters)):
        for rc in range(len(multiDepthClusters)):
            if(calcDist(truthClusters[tc], multiDepthClusters[rc]) < 15 and not(truthClusters[tc].matched)):
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
