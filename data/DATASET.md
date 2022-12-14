# Human3.6M

Fot dataset instructions on Human3.6M, see the main [README](../README.md#dataset ).

# AMASS
Pre-training on AMASS requires the conversion of the SMPL+H mesh representation in AMASS to a Human3.6M compatible joint representation.
We created a standalone [GitHub repository](https://github.com/goldbricklemon/AMASS-to-3DHPE) for this conversion step.
Follow the installation instructions from this repository to convert the individual datasets within AMASS. 
Each dataset will be converted to a file `[DATASET_NAME].npz`, with the common `.npz` dataset format used for Human3.6M.
In this paper, we converted and used the following datasets from AMASS:

  * CMU.npz
  * DanceDB.npz
  * MPILimits.npz
  * TotalCapture.npz
  * EyesJapanDataset.npz
  * HUMAN4D.npz
  * KIT.npz
  * BMLhandball.npz
  * BMLmovi.npz
  * BMLrub.npz
  * EKUT.npz
  * TCDhandMocap.npz
  * ACCAD.npz
  * Transitionsmocap.npz
  * MPIHDM05.npz
  * SFU.npz
  * MPImosh.npz

Create the directory `data/amass` and place all these files there. 