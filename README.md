# CElegans

---------------------------------------------------------
## In 'ARD':

*Single ARD*: Implements regular ARD on toy data

*Group ARD*: Implements Group ARD on toy data

---------------------------------------------------------
## In 'Data Analysis':

*AnimateCenterline*: Generate animation of centerline across all time

*Eigenworms*: Computes PCA on worm centerline data

*GCaMPSlopeResidualDeconvolve*: GCaMP and RFP signal: Read in data, discard NaN values, then calculate slopes of different neurons, residuals, and deconvolve to get neural activity

*GFPSlopeResidualDeconvolve*: GFP and RFP signal: Read in data, discard NaN values, then calculate slopes of different neurons, residuals, and deconvolve to get neural activity

---------------------------------------------------------
## In 'Data Processing':

*Consolidate\_Information*: pulls together all the different kinds of data provided by the Leifer lab to create a single .npz files for each individiual worms containing all its relevant info. Results stored in 'SharedData' folder

*ComputeCenterlineAngles*: computes centerline angles and stores those, in addition to their projections onto eig_basis and all data stored in Consolidate\_Information, into a new file called WormAngles_.npz in the 'SharedData' folder
---------------------------------------------------------
## In 'Images':

- relevant images we want to save

---------------------------------------------------------
## In 'Neuron Simulation':

*Full\_Neuron\_Simulation*: Generates neuronal activity from toy network and tries to recover connection weights with Group ARD

---------------------------------------------------------
## In 'SharedData':

- Reserved for 'final' data, i.e. data that should not be modified, only copied and used for further analysis
- Current holds the output of *Consolidated\_Information* which consolidates all the information about each of our different worms. Thus, there are three .npz files for each of our three worms.
