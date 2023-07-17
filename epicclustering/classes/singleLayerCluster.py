import numpy as np

class singleLayerCluster:
    def __init__(self, event, layer, hitIndices):
        self.event = event
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
        
    def calculateCluster(self):
        hits = self.hitIndices
        energies = np.multiply(self.hitFracs, self.event.tower_LFHCAL_E[hits])
        totalE = sum(energies)
        self.energy = totalE
        self.posx = sum(np.multiply(energies, self.event.tower_LFHCAL_posx[hits]))/totalE
        self.posy = sum(np.multiply(energies, self.event.tower_LFHCAL_posy[hits]))/totalE
        self.posz = sum(np.multiply(energies, self.event.tower_LFHCAL_posz[hits]))/totalE
            
    def dist(self, h):
        x1 = self.event.tower_LFHCAL_posx[h]
        y1 = self.event.tower_LFHCAL_posy[h]
        x2 = self.posx
        y2 = self.posy
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
