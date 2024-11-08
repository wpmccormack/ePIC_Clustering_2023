import uproot
import numpy as np

class truthCluster:
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

        
