# ePIC_Clustering_2023

Code for HCAL Clustering in ePIC detector

Numpy, uproot, and ROOT are needed.  Some examples are included based on jupyter notebooks for visualization

Throughout, I refer to non-ML based clustering as "generic"
This approach is based on the clustering algorithm used by the CMS experiment (see section 3.4 of [1706.04965](https://arxiv.org/pdf/1706.04965.pdf)).  Like the default CMS algorithm, it clusters each calorimeter layer separately before combining information from different layers.  This is not necessarily the optimal approach (rather it is due to historical readout characteristics of the CMS HCAL)

## The algorithm

The clustering algorithm here consists of several components.

1. Creating an event-level object that contains hit information and some truth information from the event.  Allows for better memory efficiency I think
   * Note: within the event, there are several numpy arrays carrying information about hit energy, position, cell index, etc.  Each of these arrays are indexed in the same way, such that an index value corresponds to a unique hit
   * Each cell has a unique ix, iy, and iz index.  A dictionary is created where the key is a tuple of (iz,iy,iz) and the value is the corresponding hit index of the hit in the cell.  This makes it easier to check if a hit has neighboring hits and what the property of those hits are
2. The hits (i.e. the hit array arguments) are sorted based on hit energy.  Each particle shower will have a highest-energy cell, which tends to be in the center of the shower.  The shower extends out radially with a roughly Gaussian die-off from the center.
   * This Gaussianized shape approximation is used several times throughout the clustering algorithm
3. "Seed" hits are found.  These are meant to correspond to the highest-energy cells in particle showers.  To be a seed, the cell must have energy over a certain threshold (set to 0.4 GeV here), and must not be within a certain distance (here set to 15 cm) of another seed or have smaller energy than would be expected from the Gaussian energy die-off approximation of a higher energy cell.  Seeds are found on a layer-by-layer basis.  A typical particle will create multiple seeds (in different layers) that roughly correspond to its trajectory through the calorimeter
4. So-called "superClusters" are created in a layer-by-layer way.  Super clusters start with the highest energy hits and expand outward through adjacent hits (but will only contain hits from one layer).  A super cluster can contain multiple seeds.  In this case, the super cluster will be split in a later step.  The super-cluster expansion step is a recursive algorithm, but each hit can only be used once
5. SuperClusters are combined in a single layer if they are very close.  This step is mostly due to current oddities in the calorimeter indexing.  It is not really necessary, though in practice, there can be discontinuities in particles showers within a calorimeter.  Discontinuities should only affect the fringe of a shower, and should only involve low-energy deposits (ie in the region where stochastic shower fluctuations can cause missed readouts for cells)
6. SuperClusters are split into "regular" clusters.  If a superCluster contains only one "seed", it is not split.  If a superCluster contains multiple seeds, then it is split (with the number of "regular" clusters corresponding to the number of seeds).  In the current implementations, the (two or more) clusters resulting from the superCluster split will contain all of the hits of the original cluster (other than the seeds used for other clusters), but with a "fraction" of the energy determined by the distance of the cell to the different seeds in the superCluster.
   * This split could be improved by later implementing a "pruning" step, which would remove hits that are too far from a seed or have too low of a fraction (or by preventing those hits from being added in the first place)
7. The individual "regular" clusters from different layers are then combined to create "multiDepth" clusters.  Each multiDepth cluster should contain either 0 or 1 cluster from each layer.  In the current implmentation, multiDepth clusters are started from layer 0, then layer 1 is checked for matching clusters, then layer 2, and so on.  If a shower doesn't leave energy in layer 0, that is ok, as a new multiDepth cluster is created for each "regular" cluster that isn't matched to an existing multiDepth cluster.  The "matching" is performed based on inter-cluster distance.  A technical detail here is that particles closer to the outer edges of the calorimeter are traveling at steeper angles through the calorimeter, meaning that they have greater layer-to-layer displacement.  For example, a particle that hits layer 0 at x = 200 will hit layer 1 at approximately x = 210, layer 2 at x = 220, and so on.  Thus the matching is performed based on an extrapolated position where the x-z slope is taken to be pos_x/20 and the y-z slope is taken to be pos_y/20.


There is a also a clustering algorithm based on truth-level information included.  For each distict truth-level particle ID in the event, information for that particle is extracted by using the cell energy and position information that the particle deposits energy in, as well as the fraction of the total energy that the particle in question deposited (current files store information for up to 4 particles per cell, unless I'm misunderstanding how the indexing is performed).  Having calculated the energy and position of each particle, I then create "truth" Clusters.  If two truth clusters are too close to each other or have too great a disparity in energy (e.g. a low energy particle next to a high energy one), then I combine their truth clusters, as they would be very difficult to distinguish in reconstruction.

## The "superEvent"

superEvent.py contains the "superEvent" class, which will full process a single event, running truth clustering and generic

## Components of clustering

The eventContainer.py, truthCluster.py, superCluster.py, singleLayerCluster.py, and multiDepthCluster.py macros all contain classes for different types of cluster (or event) objects.  The helperV2.py file contains helper functions that can be used to run clustering


## Important note for ML-based clustering schemes

An ML-based clustering method might return the set of hits that should be used in a single cluster (or rather, sets of hits that correspond to different clusters).  One can create a cluster from the hits in an event using the exampleMLBasedCluster class that I've provided.  Here, you would just need to create an eventContainer class for the event in question, and then initialize an exampleMLBasedCluster object with the hits in question for the given cluster (as a collection of the hit array indices).  If the ML model somehow returns hit fractions that should be used in a cluster, then that can be included too.