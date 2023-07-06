import numpy as np

class eventContainer:
    def __init__(self, evnum, branches):
        self.evnum = evnum
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

        if(self.tower_LFHCAL_N < 1):
            return
        
        self.args = np.argsort(-self.tower_LFHCAL_E)

        self.seeds = self.findSeeds()

        #This makes a dictionary of hits (ie their indices) by cell indices (ix, iy, iz).
        #Makes it easier to search for adjacent this
        self.recHitContainer()

        #Use a mask array so that we don't use hits in multiple "super clusters".
        #Can change hit energy threshold to speed up processing
        self.mask = list(self.tower_LFHCAL_E>0.00)

        

    def findSeeds(self):
        seeds = []
        for a in self.args:
            #Hit with energy below 0.2 cannot be a seed.
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


    def calcRawDist(self, a, s):
        x1 = self.tower_LFHCAL_posx[a]
        y1 = self.tower_LFHCAL_posy[a]
        x2 = self.tower_LFHCAL_posx[s]
        y2 = self.tower_LFHCAL_posy[s]
        return np.sqrt(np.power(x1-x2,2)+np.power(y1-y2,2))
