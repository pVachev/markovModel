import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from hmmlearn import hmm 

np.random.seed(42)

n_states = 4
seed = 7


def main():


    df = pd.read_csv("GSPC_BND.csv",
                     header=0,
                     names=["Date",
                            "BND",
                            "GSPC",
                            "LogReturnBND",
                            "LogReturnGSPC"],
                            parse_dates=True,
                            index_col=0)
    
    df_spy = pd.read_csv("SPY.csv", 
                         skiprows=[0,1],
                         usecols=[0,1],
                         header=0,
                         names=["Date", "SPY"],
                         parse_dates=True,
                         index_col=0)
    
    df_spy["LogReturnSPY"] = np.log(df_spy["SPY"].shift(1) / df_spy["SPY"])

 

    df_spy.drop(columns=["SPY"], inplace=True)

    # print(df_spy)
    
    # print(df_spy)
    
    df = pd.merge(left=df,
                  right=df_spy,
                  how='inner',
                  on="Date")
    
    df = df.resample("ME").sum()

    
    
    cols = ["LogReturnBND", "LogReturnSPY"]
    df = df[cols]

    df = df.to_numpy()


    model = hmm.GaussianHMM(n_components=n_states,
                            covariance_type="full",
                            n_iter=2000,
                            seed=7)
    
    model.fit(df)

    states = model.predict(df)
    probs = model.predict_proba(df)

    df["state"] = states
    for k in range(n_states):
        df[f"p_state_{k}"] = probs[:, k]


    g = df.groupby("state")[cols]
    out = pd.concat(
        [
            g.mean().add_prefix("mean_"),
            g.std(ddof=1).add_prefix("std_"),
            df.groupby("state").size().rename("n_obs"),
        ],
        axis=1
    )
    # quick “vol-first” ordering (use first column as reference)




    


    # print(df)


    # df_m = df[["LogReturnBND", "LogReturnGSPC"]]
    # df_m = df_m.resample("ME").sum()

    # print(df_m.head())

    



    # model = hmm.GaussianHMM(n_components=2,
    #                         )

    

if __name__ == "__main__":
    main()