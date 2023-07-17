import numpy as np

class superCluster:
    def __init__(self, event, layer):
        self.event = event
        self.hitIndices = []
        self.posx = -500
        self.posy = -500
        self.posz = -500
        self.iz = layer

    def calculateCluster(self):
        hits = self.hitIndices
        energies = self.event.tower_LFHCAL_E[hits]
        totalE = sum(energies)
        self.energy = totalE
        self.posx = sum(np.multiply(energies, self.event.tower_LFHCAL_posx[hits]))/totalE
        self.posy = sum(np.multiply(energies, self.event.tower_LFHCAL_posy[hits]))/totalE
        self.posz = sum(np.multiply(energies, self.event.tower_LFHCAL_posz[hits]))/totalE
        
    def appendHit(self, hitInd):
        self.hitIndices.append(hitInd)
        
    def appendListOfHits(self, hitInd):
        self.hitIndices = self.hitIndices+hitInd
