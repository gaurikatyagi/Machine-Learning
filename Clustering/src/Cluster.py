import numpy as np
import os

def read_csv(file_path):
    """
    First we will read in the csv file as a rec-array and transform the text fields to a numerically coded identifiers.
    This rec_array will then be converted to a numpy array.
    This function reads the csv file in the root folder and returns a numpy array of the data in order to perform clustering
    :param file_path: strinng variable- path to where the csv file is present
    :return: numpy array- data in csv
    """
    csvfile = open(file_path, "r")
    #removing mouse id as the id might influence clustering. Creating a bias if the mice were given a id according to their
    #class. We only want clustering on the basis of proteins.
    heterogeneous_data = np.genfromtxt(csvfile, delimiter = ",", names = True, usecols = range(1, 82),
            # converters = {31:make_date},
            dtype = [("DYRK1A_N", np.float64), ("ITSN1_N", np.float64),
                     ("BDNF_N", np.float64), ("NR1_N", np.float64), ("NR2A_N", np.float64),("pAKT_N", np.float64),
                     ("pBRAF_N", np.float64), ("pCAMKII_N", np.float64), ("pCREB_N", np.float64),
                     ("pELK_N", np.float64), ("pERK_N", np.float64), ("pJNK_N", np.float64), ("PKCA_N", np.float64),
                     ("pMEK_N", np.float64), ("pNR1_N", np.float64), ("pNR2A_N", np.float64), ("pNR2B_N", np.float64),
                     ("pPKCAB_N", np.float64),("pRSK_N", np.float64), ("AKT_N", np.float64), ("BRAF_N", np.float64),
                     ("CAMKII_N", np.float64), ("CREB_N", np.float64), ("ELK_N", np.float64), ("ERK_N", np.float64),
                     ("GSK3B_N", np.float64), ("JNK_N", np.float64), ("MEK_N", np.float64), ("TRKA_N", np.float64),
                     ("RSK_N", np.float64), ("APP_N", np.float64), ("Bcatenin_N", np.float64), ("SOD1_N", np.float64),
                     ("MTOR_N", np.float64), ("P38_N", np.float64), ("pMTOR_N", np.float64), ("DSCR1_N", np.float64),
                     ("AMPKA_N", np.float64), ("NR2B_N", np.float64), ("pNUMB_N", np.float64), ("RAPTOR_N", np.float64),
                     ("TIAM1_N", np.float64), ("pP70S6_N", np.float64), ("NUMB_N", np.float64), ("P70S6_N", np.float64),
                     ("pGSK3B_N", np.float64), ("pPKCG_N", np.float64), ("CDK5_N", np.float64), ("S6_N", np.float64),
                     ("ADARB1_N", np.float64), ("AcetylH3K9_N", np.float64), ("RRP1_N", np.float64), ("BAX_N", np.float64),
                     ("ARC_N", np.float64), ("ERBB4_N", np.float64), ("nNOS_N", np.float64), ("Tau_N", np.float64),
                     ("GFAP_N", np.float64), ("GluR3_N", np.float64), ("GluR4_N", np.float64), ("IL1B_N", np.float64),
                     ("P3525_N", np.float64), ("pCASP9_N", np.float64), ("PSD95_N", np.float64), ("SNCA_N", np.float64),
                     ("Ubiquitin_N", np.float64), ("pGSK3B_Tyr216_N", np.float64), ("SHH_N", np.float64),
                     ("BAD_N", np.float64), ("BCL2_N", np.float64), ("pS6_N", np.float64), ("pCFOS_N", np.float64),
                     ("SYP_N", np.float64), ("H3AcK18_N", np.float64), ("EGR1_N", np.float64), ("H3MeK4_N", np.float64),
                     ("CaNA_N", np.float64), ("Genotype", "|S11"), ("Treatment", "|S11"), ("Behavior", "|S11"),
                     ("class", "|S11")]
                               )
    print heterogeneous_data


if __name__ == "__main__":
    directory_path =  os.path.join(os.path.dirname( os.path.dirname(os.path.abspath( __file__ ))))
    file_name = "Data_Cortex_Nuclear.csv"
    file_path = os.path.join(directory_path, file_name)
    data = read_csv(file_path)
