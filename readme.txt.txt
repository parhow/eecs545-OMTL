The code in this folder is for use in conjunction
 with the project on online multtitask learning (OMTL).

Authors: Sehwan Chung, Parker Howard, Gautam Thakur


LandmineData.zip contains the "landmine" dataset that
was used to produce the 2 real sets for evaluation of code.
This dataset is defined as the real set.

realData_baselines is used to get results of baseline
	methods for the real data (CMTL, BatchOpt)
realData_OMTL is used to obtain results for the different
	OMTL methods (LogDet, von-Neumann, Covariance)
synthetic_CTML is used for results of CTML applied to 
	the synthetic dataset.
synthetic_LogDet is used for results of the OMTL LogDet
	algorithm on the synthetic dataset.


The realData and and synthetic Python files do preprocssing
of the respective data type and importing of files.