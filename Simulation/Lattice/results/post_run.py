# (run where spectrum.csv is available)
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv("spectrum.csv")
eigs = df['lambda'].to_numpy()
tlist = np.logspace(-4,2,200)
P = np.array([np.mean(np.exp(-eigs * t)) for t in tlist])
results=[]
for tmin in np.logspace(-4,-1,8):
  for tmax in np.logspace(-1,1,8):
    if tmax <= tmin: continue
    mask=(tlist>=tmin)&(tlist<=tmax)&(P>0)
    if mask.sum()<5: continue
    X=np.log(tlist[mask]).reshape(-1,1); y=np.log(P[mask])
    reg=LinearRegression().fit(X,y)
    d_s=-2*reg.coef_[0]; r2=reg.score(X,y)
    results.append((tmin,tmax,d_s,r2))
pd.DataFrame(results,columns=['tmin','tmax','d_s','R2']).sort_values('R2',ascending=False).head(20)
