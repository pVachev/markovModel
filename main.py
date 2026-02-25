import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from hmmlearn import hmm 

np.random.seed(42)


def main():


    df = pd.read_csv("GSPC_BND.csv",
                     header=0,
                     names=["Date",
                            "BND",
                            "GSPC",
                            "LogReturnBND",
                            "LogReturnGSPC"],
                            parse_dates=True)

    # print(df)

    print(df)

    # model = hmm.GaussianHMM(n_components=2,
    #                         )

    

if __name__ == "__main__":
    main()