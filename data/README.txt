Each data contains 4 variables: G, D_new, DU and DO

G contains all the within-layer networks.

DU contains the complete cross-layer dependencies across the layers.

DO contains the observed cross-layer dependencies, which is a subset of DU.

D_new is a symmetric matrix that describe the dependency across the layers. The non-zero number in the matrix gives the index of the dependency matrix in DU and DO.