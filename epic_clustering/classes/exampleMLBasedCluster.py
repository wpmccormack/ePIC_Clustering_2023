import numpy as np

class exampleMLBasedCluster:
    def __init__(self, event, hitIndices=[], hitFracs=[]):
        self.event = event
        #self.layerSet = set([SLC.iz])
        self.hitIndices = hitIndices
        if(len(hitFracs)>0):
            self.hitFracs = hitFracs
        else:
            self.hitFracs = [1]*len(hitIndices)
        self.truthMatch = -1

    def calculateCluster(self):
        hits = self.hitIndices
        energies = np.multiply(self.hitFracs, self.event.tower_LFHCAL_E[hits])
        totalE = sum(energies)
        self.energy = totalE
        self.posx = sum(np.multiply(energies, self.event.tower_LFHCAL_posx[hits]))/totalE
        self.posy = sum(np.multiply(energies, self.event.tower_LFHCAL_posy[hits]))/totalE
        self.posz = sum(np.multiply(energies, self.event.tower_LFHCAL_posz[hits]))/totalE
        self.layerSet = set(self.event.tower_LFHCAL_iz[hits])

    
    def dist(self, h):
        x1 = self.event.tower_LFHCAL_posx[h]
        y1 = self.event.tower_LFHCAL_posy[h]
        x2 = self.posx
        y2 = self.posy
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
    
    def distClust(self, slc):
        x1 = slc.posx
        y1 = slc.posy
        x2 = self.posx
        y2 = self.posy
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
    
    def distClustLayer(self, slc, layer):
        x1 = slc.posx
        y1 = slc.posy
        x2 = self.singleLayerClusters[layer].posx
        y2 = self.singleLayerClusters[layer].posy
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
    
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
        
