import uproot
import numpy as np

#Author: Patrick McCormack
#Date: June 30, 2023
#Email: wmccorma@mit.edu

#This is a class that does the whole generic clustering step (and truth-based clustering) in one go
#This is a somewhat unwieldy class, but I think this is a *more* memory efficient implementation in python (since I can't use pointers and references as I normally would in C++)
#####I'm not going to claim that it is the *most* efficient implementation, since I'm not good enough of a programmer to make that claim

#Note, you'll see many instance of a formula in the form:
#np.exp(-0.5 * np.power(distance/10.,2))
#Throughout, I roughly assume a shower sigma of 10.
#Right now, this is lazily hard-coded, and chosen based on a qualitative study of truth clusters, rather than a systematic, quantitative study

class superEvent:
    def __init__(self, evnum, branches):
        self.evnum = evnum        
        self.tower_LFHCAL_N = np.asarray(branches['tower_LFHCAL_N'][self.evnum])
        self.tower_LFHCAL_E = np.asarray(branches['tower_LFHCAL_E'][self.evnum])
        self.tower_LFHCAL_ix = np.asarray(branches['tower_LFHCAL_ix'][self.evnum])
        self.tower_LFHCAL_iy = np.asarray(branches['tower_LFHCAL_iy'][self.evnum])
        self.tower_LFHCAL_iz = np.asarray(branches['tower_LFHCAL_iz'][self.evnum])
        self.tower_LFHCAL_posx = np.asarray(branches['tower_LFHCAL_posx'][self.evnum])
        self.tower_LFHCAL_posy = np.asarray(branches['tower_LFHCAL_posy'][self.evnum])
        self.tower_LFHCAL_posz = np.asarray(branches['tower_LFHCAL_posz'][self.evnum])
        self.tower_LFHCAL_NContributions = np.asarray(branches['tower_LFHCAL_NContributions'][self.evnum])
        self.tower_LFHCAL_trueID1 = np.asarray(branches['tower_LFHCAL_trueID1'][self.evnum])
        self.tower_LFHCAL_trueID2 = np.asarray(branches['tower_LFHCAL_trueID2'][self.evnum])
        self.tower_LFHCAL_trueID3 = np.asarray(branches['tower_LFHCAL_trueID3'][self.evnum])
        self.tower_LFHCAL_trueID4 = np.asarray(branches['tower_LFHCAL_trueID4'][self.evnum])
        self.tower_LFHCAL_trueEfrac1 = np.asarray(branches['tower_LFHCAL_trueEfrac1'][self.evnum])
        self.tower_LFHCAL_trueEfrac2 = np.asarray(branches['tower_LFHCAL_trueEfrac2'][self.evnum])
        self.tower_LFHCAL_trueEfrac3 = np.asarray(branches['tower_LFHCAL_trueEfrac3'][self.evnum])
        self.tower_LFHCAL_trueEfrac4 = np.asarray(branches['tower_LFHCAL_trueEfrac4'][self.evnum])
        
        self.indexDict_Layered = {}
        self.truthClusters = []
        self.listOfSCs = []
        self.combinedListOfSCs = []
        self.allClusters = []
        self.layeredSLCs = []
        self.multiDepthClusters = []
        self.matchedClusters = []
        self.diffs = []

        if(self.tower_LFHCAL_N < 1):
            return

        #Overarching principle: hits correspond to their index
        #Sort hits by energy.  Strategy: expand clusters out from highest energy hits
        self.args = np.argsort(-self.tower_LFHCAL_E)
        
        #We need to find our seeds
        self.seeds = self.findSeeds()

        #This makes a dictionary of hits (ie their indices) by cell indices (ix, iy, iz).
        #Makes it easier to search for adjacent this
        self.recHitContainer()

        #Make the truth clusters in an event
        self.makeTruthClusters()
        self.truthClusters.sort(key=lambda c: -c.energy)

        #Use a mask array so that we don't use hits in multiple "super clusters".
        #Can change hit energy threshold to speed up processing
        self.mask = list(self.tower_LFHCAL_E>0.00)

        #SuperClusters are expanded outward through adjacent cells.
        #SuperClusters live in a single layer
        #A SuperCluster can contain multiple seeds
        self.makeSuperClusters()

        #Combine superClusters if they're very close.  (probably unnecessary)
        self.combineSuperClustersInLayer()

        #Split superClusters into single layer "regular clusters" that only contain 1 seed
        self.makeAllClusters()
        self.allClusters.sort(key=lambda c: -c.energy)

        #Make a structured array of "regular" clusters arranged by layer and energy.
        #We will want to combine clusters from different layers into single clusters
        self.izs = [c.iz for c in self.allClusters]
        self.makeLayeredClusters()

        #This handles the combination step
        self.makeMultiDepthClusters()
        
        #This executes a reco-cluster (multiDepthClusters) to truthCluster matching scheme
        self.doClusterMatching()

        #Puts together an array of (truth energy - reco energy)/truth energy for matched truth clusters
        self.findMatchedDiffs()
        
        
    def findSeeds(self):
        seeds = []
        for a in self.args:
            #Hit with energy below 0.4 cannot be a seed.
            #This was a somewhat arbitrary choice.
            if self.tower_LFHCAL_E[a] < 0.4:
                #a comes from a sorted array (sorted based on energy), so no subsequent a's would pass the threshold
                break
            newSeed = True
            for s in seeds:
                if(not self.tower_LFHCAL_iz[a] == self.tower_LFHCAL_iz[s]):
                    continue
                #Check raw distance between seeds and whether the potentially different seed has high enough relative energy, assuming shower sigma of 10 here.
                if(self.calcRawDist(a,s) < 15. or self.tower_LFHCAL_E[a]/self.tower_LFHCAL_E[s] < np.exp(-0.5 * np.power(self.calcRawDist(a,s)/10.,2))):
                    newSeed = False
            if(newSeed):
                seeds.append(a)

        return seeds
    
    
    def recHitContainer(self):
        for a in self.args:
            self.indexDict_Layered[(self.tower_LFHCAL_ix[a], self.tower_LFHCAL_iy[a], self.tower_LFHCAL_iz[a])] = a

    
    def makeTruthClusters(self):
        #This is more verbose than I would like, but the variables of interest (ID and Frac) have different, hard-coded names
        tmpTCs = []
        for i in range(max(self.tower_LFHCAL_trueID1)+1):
            hits1 = np.argwhere(self.tower_LFHCAL_trueID1==i)
            if(len(hits1)>1 or len(hits1)==0):
                hits1 = np.squeeze(hits1)
            else:
                hits1 = hits1[0]
            hits2 = np.argwhere(self.tower_LFHCAL_trueID2==i)
            if(len(hits2)>1 or len(hits2)==0):
                hits2 = np.squeeze(hits2)
            else:
                hits2 = hits2[0]
            hits3 = np.argwhere(self.tower_LFHCAL_trueID3==i)
            if(len(hits3)>1 or len(hits3)==0):
                hits3 = np.squeeze(hits3)
            else:
                hits3 = hits3[0]
            hits4 = np.argwhere(self.tower_LFHCAL_trueID4==i)
            if(len(hits4)>1 or len(hits4)==0):
                hits4 = np.squeeze(hits4)
            else:
                hits4 = hits4[0]
            
            e1 = self.tower_LFHCAL_E[hits1]
            e2 = self.tower_LFHCAL_E[hits2]
            e3 = self.tower_LFHCAL_E[hits3]
            e4 = self.tower_LFHCAL_E[hits4]
            f1 = self.tower_LFHCAL_trueEfrac1[hits1]
            f2 = self.tower_LFHCAL_trueEfrac2[hits2]
            f3 = self.tower_LFHCAL_trueEfrac3[hits3]
            f4 = self.tower_LFHCAL_trueEfrac4[hits4]
            
            tmpE = sum(np.multiply(e1,f1)) + sum(np.multiply(e2,f2)) + sum(np.multiply(e3,f3)) + sum(np.multiply(e4,f4))
            if(tmpE > 0):
                tmpX = sum(np.multiply(np.multiply(e1,f1), self.tower_LFHCAL_posx[hits1]))
                tmpX += sum(np.multiply(np.multiply(e2,f2), self.tower_LFHCAL_posx[hits2]))
                tmpX += sum(np.multiply(np.multiply(e3,f4), self.tower_LFHCAL_posx[hits3]))
                tmpX += sum(np.multiply(np.multiply(e3,f4), self.tower_LFHCAL_posx[hits4]))
                tmpX = tmpX/tmpE
                tmpY = sum(np.multiply(np.multiply(e1,f1), self.tower_LFHCAL_posy[hits1]))
                tmpY += sum(np.multiply(np.multiply(e2,f2), self.tower_LFHCAL_posy[hits2]))
                tmpY += sum(np.multiply(np.multiply(e3,f4), self.tower_LFHCAL_posy[hits3]))
                tmpY += sum(np.multiply(np.multiply(e3,f4), self.tower_LFHCAL_posy[hits4]))
                tmpY = tmpY/tmpE
                tmpZ = sum(np.multiply(np.multiply(e1,f1), self.tower_LFHCAL_posz[hits1]))
                tmpZ += sum(np.multiply(np.multiply(e2,f2), self.tower_LFHCAL_posz[hits2]))
                tmpZ += sum(np.multiply(np.multiply(e3,f4), self.tower_LFHCAL_posz[hits3]))
                tmpZ += sum(np.multiply(np.multiply(e3,f4), self.tower_LFHCAL_posz[hits4]))
                tmpZ = tmpZ/tmpE
                tmpTCs.append(self.innerTruthCluster(i, tmpE, tmpX, tmpY, tmpZ, list(hits1)+list(hits2)+list(hits3)+list(hits4), list(f1)+list(f2)+list(f3)+list(f4)))
            
        tmpTCs.sort(key=lambda c: -c.energy)

        #I combine truth particles if they're too close
        #Clusters are stochastic enough that discovering a small "sub-cluster" within a larger cluster is very hard from a reconstruction point of view (though perhaps not impossible)
        keptTCs = [0]
        for tc2 in range(1,len(tmpTCs)):
            absorbed = False
            for tc1 in range(tc2):
                tcDist = self.calcDist(tmpTCs[tc1], tmpTCs[tc2])
                if(tmpTCs[tc2].energy < tmpTCs[tc1].energy*np.exp(-0.5 * np.power(tcDist/10.,2))):
                    tmpTCs[tc1].combineTwoTruthClusters(tmpTCs[tc2])
                    absorbed = True
                    break
            if(not absorbed and tmpTCs[tc2].energy > 0.5):
                keptTCs.append(tc2)

        for k in keptTCs:
            self.truthClusters.append(tmpTCs[k])

                            
    def makeSuperClusters(self):
        #Hits are sorted.
        #expandCluster is a recursive algorithm that will mask hits as they're used in superClusters
        for a in self.args:
            if(not self.mask[a]):
                continue
            tmpSC = self.innerSuperCluster(self.tower_LFHCAL_iz[a])
            self.mask[a] = False
            self.expandCluster(a, tmpSC)
            self.calculateSuperCluster(tmpSC)
            self.listOfSCs.append(tmpSC)
            
        self.listOfSCs.sort(key=lambda sc: -sc.energy)
            
    
    def expandCluster(self, argInQuestion, superCluster):

        superCluster.appendHit(argInQuestion)
        currX = self.tower_LFHCAL_ix[argInQuestion]
        currY = self.tower_LFHCAL_iy[argInQuestion]
        currZ = self.tower_LFHCAL_iz[argInQuestion]

        #Here I scan a grid of adjacent indices from two indices to the left to two to the right, and from 2 down to two up
        #This scan is probably too large, but during development, the indexing isn't great.
        #If you're counting, this scan checks 24 other indices (5x2 grid on the left, 5x2 grid on the right, and 2x1 up and 2x1 down)
        #In CMS for example, the default is only 4 other indices: one left, one right, one up, and one down, but I trust the indexing there
        for i in range(-2, 3):
            for j in range(-2, 3):
                if(not (i==0 and j==0) and currX+i >=0 and currY+j >=0):
                    testTuple = (currX+i, currY+j, currZ)
                    if(testTuple in self.indexDict_Layered and self.mask[self.indexDict_Layered[testTuple]]):
                        neighborArg = self.indexDict_Layered[testTuple]
                        self.mask[neighborArg] = False
                        self.expandCluster(neighborArg, superCluster)
    
    
    def combineSuperClustersInLayer(self):
        for i in range(len(self.listOfSCs)):
            new = True
            for s in range(len(self.combinedListOfSCs)):
                if(not self.listOfSCs[i].iz == self.combinedListOfSCs[s].iz):
                    continue
                if(self.calcDist(self.listOfSCs[i],self.combinedListOfSCs[s]) < 20.):
                    self.combinedListOfSCs[s].appendListOfHits(self.listOfSCs[i].hitIndices)
                    self.calculateSuperCluster(self.combinedListOfSCs[s])
                    new = False
                    break
            if(new):
                self.combinedListOfSCs.append(self.listOfSCs[i])

        self.combinedListOfSCs.sort(key=lambda sc: -sc.energy)
        
        
    def makeAllClusters(self):
        #Here we're making smaller, "regular" clusters that only have one distinct seed in them.
        #Regular clusters can share hits with other regular clusters that both come from the same superCluster, but with energy fractions determined by the distance from hit to cluster center
        #I should probably add in some cluster "pruning" to remove hits that are far from the centroid or contribute only a small fraction of their energy to the cluster
        for c in self.combinedListOfSCs:
            if(c.energy < 0.1):
                break
            self.allClusters += self.splitSuperCluster(c)

    
    def splitSuperCluster(self, SC):
        arrayOfClusters = []
        seedsInSC = np.intersect1d(self.seeds, SC.hitIndices)
        numSeedsInSC = len(seedsInSC)

        if(numSeedsInSC <= 1):
            tmpCluster = self.innerSingleLayerCluster(SC.iz, SC.hitIndices)
            tmpCluster.setHitFrac([1]*len(SC.hitIndices))
            self.calculateSplitCluster(tmpCluster)
            arrayOfClusters.append(tmpCluster)
        else:
            #Split the superCluster
            hitsInSC = SC.hitIndices
            scHitMask = [True]*(max(hitsInSC)+1)
            for s in seedsInSC:
                #Initialize split clusters using the seeds in the superCluster
                tmpCluster = self.innerSingleLayerCluster(SC.iz, [s])
                tmpCluster.appendHitFrac(1)
                self.calculateSplitCluster(tmpCluster)
                arrayOfClusters.append(tmpCluster)
                scHitMask[s] = False
            for h in hitsInSC:
                if(not scHitMask[h]):
                    continue
                #Energy fraction assigment based on distance.  Using shower sigma of 10 now
                sumEnergyContrib = 0.
                for c in arrayOfClusters:
                    sumEnergyContrib += c.energy*np.exp(-0.5 * np.power(self.calcDistClusterToArg(c,h)/10.,2))
                for c in arrayOfClusters:
                    c.appendHit(h)
                    c.appendHitFrac(c.energy*np.exp(-0.5 * np.power(self.calcDistClusterToArg(c,h)/10.,2))/sumEnergyContrib)
                    self.calculateSplitCluster(c)

        return arrayOfClusters
    
    
    def makeLayeredClusters(self):
        for i in range(7):
            if(len(np.argwhere(np.asarray(self.izs)==i)) > 1):
                inlayer = list(np.squeeze(np.argwhere(np.asarray(self.izs)==i)))
                self.layeredSLCs.append(list(np.asarray(self.allClusters)[inlayer]))
            elif(len(np.argwhere(np.asarray(self.izs)==i)) == 1):
                inlayer = np.squeeze(np.argwhere(np.asarray(self.izs)==i))
                self.layeredSLCs.append([np.asarray(self.allClusters)[inlayer]])
            else:
                self.layeredSLCs.append([])

            self.layeredSLCs[i].sort(key=lambda c: -c.energy)
    
    
    def makeMultiDepthClusters(self):
        for i in range(7):
            for c in self.layeredSLCs[i]:
                used = False
                for m in self.multiDepthClusters:
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
                                    self.calculateMultiDepthCluster(m)
                                    used = True
                if(not used):
                    tmpMDC = self.innerMultiDepthCluster(c)
                    self.calculateMultiDepthCluster(tmpMDC)
                    self.multiDepthClusters.append(tmpMDC)

        self.multiDepthClusters.sort(key=lambda c: -c.energy)
        
        
    def doClusterMatching(self):
        #initialization, just in case
        for tc in self.truthClusters:
            tc.matched = False
        for rc in self.multiDepthClusters:
            rc.truthMatch = -1 #dummy value, since truthMatch points to the truth particle's ID value
        for tc in range(len(self.truthClusters)):
            for rc in range(len(self.multiDepthClusters)):
                if(self.calcDist(self.truthClusters[tc], self.multiDepthClusters[rc]) < 15
                   and not(self.truthClusters[tc].matched)
                   and (self.multiDepthClusters[rc].truthMatch < 0)):
                    self.multiDepthClusters[rc].truthMatch = self.truthClusters[tc].label
                    self.truthClusters[tc].matched = True
                    self.matchedClusters.append((tc,rc))
                    break
            if(not(self.truthClusters[tc].matched)):
                self.matchedClusters.append((tc,None))
        for rc in range(len(self.multiDepthClusters)):
            if(self.multiDepthClusters[rc].truthMatch<0):
                self.matchedClusters.append((None,rc))
    
    def findMatchedDiffs(self):
        for mc in self.matchedClusters:
            if(not mc[0] is None and not mc[1] is None):
                self.diffs.append((self.truthClusters[mc[0]].energy - self.multiDepthClusters[mc[1]].energy)/self.truthClusters[mc[0]].energy)

    
    class innerTruthCluster:
        def __init__(self, label, energy, posx, posy, posz, hitIndices, hitFracs):
            self.hitIndices = hitIndices
            self.hitFracs = hitFracs
            self.label = label
            self.energy = energy
            self.posx = posx
            self.posy = posy
            self.posz = posz
            self.matched = False
            
        def combineTwoTruthClusters(self, TC):
            self.hitIndices = self.hitIndices + TC.hitIndices
            self.hitFracs = self.hitFracs + TC.hitFracs
            self.label = self.label
            self.posx = (self.posx*self.energy + TC.posx*TC.energy)/(self.energy + TC.energy)
            self.posy = (self.posy*self.energy + TC.posy*TC.energy)/(self.energy + TC.energy)
            self.posz = (self.posz*self.energy + TC.posz*TC.energy)/(self.energy + TC.energy)
            self.energy = self.energy + TC.energy
            self.matched = self.matched or TC.matched
            
            
    class innerSuperCluster:
        def __init__(self, layer):
            self.hitIndices = []
            self.posx = -500
            self.posy = -500
            self.posz = -500
            self.iz = layer
            
        def appendHit(self, hitInd):
            self.hitIndices.append(hitInd)
        def appendListOfHits(self, hitInd):
            self.hitIndices = self.hitIndices+hitInd

        
    class innerSingleLayerCluster:
        def __init__(self, layer, hitIndices):
            self.hitIndices = hitIndices
            self.hitFracs = []
            self.posx = -500
            self.posy = -500
            self.posz = -500
            self.iz = layer
            
        def appendHit(self, hitInd):
            self.hitIndices.append(hitInd)
        def appendHitFrac(self, hitFrac):
            self.hitFracs.append(hitFrac)
        def appendListOfHits(self, hitInd):
            self.hitIndices = self.hitIndices+hitInd
        def setHitFrac(self, hitFracs):
            self.hitFracs = hitFracs
        def appendListOfHitFracs(self, hitFracs):
            self.hitFracs = self.hitFracs+hitFracs
            
            
    class innerMultiDepthCluster:
        def __init__(self, SLC):
            self.layerSet = set([SLC.iz])
            self.hitIndices = SLC.hitIndices
            self.hitFracs = SLC.hitFracs
            self.singleLayerClusters = []
            for i in range(7):
                if(i == SLC.iz):
                    self.singleLayerClusters.append(SLC)
                else:
                    self.singleLayerClusters.append(None)
            self.truthMatch = -1
            
        def addSingleLayerCluster(self, SLC):
            self.hitIndices = self.hitIndices+SLC.hitIndices
            self.hitFracs = self.hitFracs+SLC.hitFracs
            self.singleLayerClusters[SLC.iz] = SLC
            
        def distClustLayerExtrapolate(self, x1, y1, layer):
            x2 = self.singleLayerClusters[layer].posx
            y2 = self.singleLayerClusters[layer].posy
            return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
            

    #Normally, I would put all of these "calculation functions" as members of their respective classes.
    #However, I'm not completely sure if this is the most memory efficient approach in python, since I want these functions to use the same energy and position arrays already stored in the superEvent class without unnecessary copies
    def calculateSuperCluster(self, SC):
        hits = SC.hitIndices
        energies = self.tower_LFHCAL_E[hits]
        totalE = sum(energies)
        SC.energy = totalE
        SC.posx = sum(np.multiply(energies, self.tower_LFHCAL_posx[hits]))/totalE
        SC.posy = sum(np.multiply(energies, self.tower_LFHCAL_posy[hits]))/totalE
        SC.posz = sum(np.multiply(energies, self.tower_LFHCAL_posz[hits]))/totalE
        
    def calculateSplitCluster(self, SC):
        hits = SC.hitIndices
        energies = np.multiply(SC.hitFracs, self.tower_LFHCAL_E[hits])
        totalE = sum(energies)
        SC.energy = totalE
        SC.posx = sum(np.multiply(energies, self.tower_LFHCAL_posx[hits]))/totalE
        SC.posy = sum(np.multiply(energies, self.tower_LFHCAL_posy[hits]))/totalE
        SC.posz = sum(np.multiply(energies, self.tower_LFHCAL_posz[hits]))/totalE
        
    def calculateMultiDepthCluster(self, MDC):
        hits = MDC.hitIndices
        energies = np.multiply(MDC.hitFracs, self.tower_LFHCAL_E[hits])
        totalE = sum(energies)
        MDC.energy = totalE
        MDC.posx = sum(np.multiply(energies, self.tower_LFHCAL_posx[hits]))/totalE
        MDC.posy = sum(np.multiply(energies, self.tower_LFHCAL_posy[hits]))/totalE
        MDC.posz = sum(np.multiply(energies, self.tower_LFHCAL_posz[hits]))/totalE
        MDC.layerSet = set(self.tower_LFHCAL_iz[MDC.hitIndices])

    #Sorry for the silly names here.  Had to come up with different ways to say the same thing
    #Normally, in C++, I would just define multiple functions with the same name and different input types, but I don't think you can do that with python
    def calcDist(self, sc1, sc2):
        return np.sqrt(np.power(sc1.posx-sc2.posx,2)+np.power(sc1.posy-sc2.posy,2))

    def calcDistRaw(self, x1, y1, x2, y2):
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))

    def calcRawDist(self, a, s):
        x1 = self.tower_LFHCAL_posx[a]
        y1 = self.tower_LFHCAL_posy[a]
        x2 = self.tower_LFHCAL_posx[s]
        y2 = self.tower_LFHCAL_posy[s]
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
    
    def calcDistClusterToArg(self, c, a):
        x1 = c.posx
        y1 = c.posy
        x2 = self.tower_LFHCAL_posx[a]
        y2 = self.tower_LFHCAL_posy[a]
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
