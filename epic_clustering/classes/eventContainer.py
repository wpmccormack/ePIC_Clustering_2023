import numpy as np
import pandas as pd

class eventContainer:
    def __init__(self, evnum, branches):
        self.evnum = evnum
        self.tower_LFHCAL_N = np.asarray(branches['tower_LFHCAL_N'][self.evnum])
        self.tower_LFHCAL_NMCParticles = np.asarray(branches['tower_LFHCAL_NMCParticles'][self.evnum])
        self.tower_LFHCAL_E = np.asarray(branches['tower_LFHCAL_E'][self.evnum])
        self.tower_LFHCAL_T = np.asarray(branches['tower_LFHCAL_T'][self.evnum])
        self.tower_LFHCAL_ix = np.asarray(branches['tower_LFHCAL_ix'][self.evnum])
        self.tower_LFHCAL_iy = np.asarray(branches['tower_LFHCAL_iy'][self.evnum])
        self.tower_LFHCAL_iz = np.asarray(branches['tower_LFHCAL_iz'][self.evnum])
        self.tower_LFHCAL_posx = np.asarray(branches['tower_LFHCAL_posx'][self.evnum])
        self.tower_LFHCAL_posy = np.asarray(branches['tower_LFHCAL_posy'][self.evnum])
        self.tower_LFHCAL_posz = np.asarray(branches['tower_LFHCAL_posz'][self.evnum])
        self.tower_LFHCAL_NContributions = np.asarray(branches['tower_LFHCAL_NContributions'][self.evnum])

        for i in range(1, 11):
            setattr(self, f'tower_LFHCAL_trueID{i}', np.asarray(branches[f'tower_LFHCAL_trueID{i}'][self.evnum]))
            setattr(self, f'tower_LFHCAL_trueEfrac{i}', np.asarray(branches[f'tower_LFHCAL_trueEfrac{i}'][self.evnum]))
            setattr(self, f'tower_LFHCAL_truePDG{i}', np.asarray(branches[f'tower_LFHCAL_truePDG{i}'][self.evnum]))

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


    # Convenience utilities for displaying the event and converting it

    def __repr__(self):
        # A multiline string that will be printed when we ask for print(eventContainer)

        return f"""---- Event #{self.evnum} ----
Number of hits: {self.tower_LFHCAL_N}
Number of unique particles: {self.tower_LFHCAL_NMCParticles}
Total energy: {self.tower_LFHCAL_E.sum()}"""
    
    def __len__(self):
        # The length of the event is the number of hits
        return self.tower_LFHCAL_N

    def __getitem__(self, key):
        # This allows us to get the value of a key by eventContainer[key]
        return getattr(self, key)
    
    def __iter__(self):
        # This allows us to iterate over the keys of the eventContainer
        """
        Returns the next tuple of (x, y, z, E, xi, yi, zi, ID1 ... ID10)
        """

        attributes_to_return = ["tower_LFHCAL_posx", "tower_LFHCAL_posy", "tower_LFHCAL_posz", "tower_LFHCAL_E", "tower_LFHCAL_ix", "tower_LFHCAL_iy", "tower_LFHCAL_iz"] + [f"tower_LFHCAL_trueID{i}" for i in range(1, 11)]
        
        for i in range(len(self)):
            yield tuple(getattr(self, attr)[i] for attr in attributes_to_return)

    def to_pandas(self):
        # This returns a pandas dataframe of the eventContainer by building a dictionary of all the columns that are "hitlike" (i.e. have length N)
        
        hitlike_columns = ["E", "ix", "iy", "iz", "posx", "posy", "posz", "NContributions", "T"] + [f"trueID{i}" for i in range(1, 11)] + [f"trueEfrac{i}" for i in range(1, 11)] + [f"truePDG{i}" for i in range(1, 11)]
        hitlike_dict = {col: getattr(self, "tower_LFHCAL_" + col) for col in hitlike_columns}

        return pd.DataFrame(hitlike_dict)