import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
import itertools

DEBUG = False
class cryptoPairsTrader():
    def __init__(self):
        self.tight=.96
        self.wide=1.04
        self.lags=5
        self.startCash=10000
        self.dataPath="dailydata/"
        self.symFile=self.dataPath+"syms.txt"
        self.dataFileExt="USD.csv"
        
    def loadData(self):
        self.syms = [sym.strip() for sym in open(self.symFile)]
        self.files = [self.dataPath+sym+self.dataFileExt for sym in self.syms]
        self.cryptos = [pd.read_csv(file) for file in self.files]
        
        self.cryptoPairs = list(itertools.combinations(self.cryptos, 2))
        self.namePairs = list(itertools.combinations(self.syms,2))
    
    def runAllPairs(self):
        cryptoPairs=self.cryptoPairs
        namePairs=self.namePairs
        self.results=pd.DataFrame()
        rets={}
        stats={}
        i=0
        for c in cryptoPairs:
            if DEBUG: print("----- Pair ",i+1," -----")
            res = {"Pair Name":namePairs[i][0]+"/"+namePairs[i][1],"Coin1":namePairs[i][0],"Coin2":namePairs[i][1]}
            rets = self.runPairsTradingStrategy(c[0],c[1],namePairs[i][0],namePairs[i][1])
            stats = self.testCointegration(c[0],c[1],namePairs[i][0],namePairs[i][1])
            res=dict(list(res.items())+list(rets.items())+list(stats.items()))
            #print(res)
            newrow = pd.DataFrame(res)
            self.results = pd.concat([self.results,newrow])
            #print(self.results)
            i+=1
        self.results.to_csv("results.csv",index=False)
            
    # When the Dickey-Fuller test statistic is very low, providing us with a low p-value. 
    # We can likely reject the null hypothesis of the presence of a unit root and conclude that we 
    # have a stationary series and hence a cointegrated pair. 
    # Our test statistic gives a p-value less than 0.05 providing evidence that cointegration exists
    # p-score less than 0.05, we reject null hypothesis of unit root, meaning no unit root, yes cointegrated
    #
    # The ADF Statistic adf[0], will tell you at which confidence level you 
    # can reject the null hypothesis.  An ADF statistic less than -3.5, the critical
    # value for 1% confidence level, or 99% probability, we can reject.  Usually a very low
    # ADF Statistic, like -4 will have a very low p-value as well.
    def testCointegration(self,coin1,coin2,sym1,sym2):
        if DEBUG: print("Cointegration test for pair ",sym1,"/",sym2,"...",sep="")
        olsres = sm.OLS(coin1["Price"],coin2["Price"]).fit()
        adf = ts.adfuller(olsres.resid)
        olsres2 = sm.OLS(coin2["Price"],coin1["Price"]).fit()
        adf2 = ts.adfuller(olsres2.resid)
        
        if DEBUG: print(sym1,"/",sym2," ADF Statistic: ",frmt4(adf[0]),sep="")
        if DEBUG: print(sym1,"/",sym2," p-value: ",frmt4(adf[1]),sep="")
        if DEBUG: print(sym1,"/",sym2," 1% Conf: ",frmt4(adf[4]['1%']),sep="")
        if DEBUG: print(sym1,"/",sym2," 5% Conf: ",frmt4(adf[4]['5%']),sep="")
        if DEBUG: print(sym1,"/",sym2," 10% Conf: ",frmt4(adf[4]['10%']),sep="")
        if DEBUG: print(sym2,"/",sym1," ADF Statistic: ",frmt4(adf2[0]),sep="")
        if DEBUG: print(sym2,"/",sym1," p-value: ",frmt4(adf2[1]),sep="")
        if DEBUG: print(sym2,"/",sym1," 1% Conf: ",frmt4(adf2[4]['1%']),sep="")
        if DEBUG: print(sym2,"/",sym1," 5% Conf: ",frmt4(adf2[4]['5%']),sep="")
        if DEBUG: print(sym2,"/",sym1," 10% Conf: ",frmt4(adf2[4]['10%']),"\n",sep="")
    
        if DEBUG: print(sym1,"Max:",frmt4(np.max(coin1["Price"])))
        if DEBUG: print(sym1,"Min:",frmt4(np.min(coin1["Price"])))
        if DEBUG: print(sym1,"Mean:",frmt4(np.mean(coin1["Price"])))
        if DEBUG: print(sym1,"Std Dev:",frmt4(np.std(coin1["Price"])))
        
        if DEBUG: print(sym2,"Max:",frmt4(np.max(coin2["Price"])))
        if DEBUG: print(sym2,"Min:",frmt4(np.min(coin2["Price"])))
        if DEBUG: print(sym2,"Mean:",frmt4(np.mean(coin2["Price"])))
        if DEBUG: print(sym2,"Std Dev:",frmt4(np.std(coin2["Price"])),"\n")
        
        return {
            "Coin1 ADF Stat":[adf[0]],
            "Coin1 p-value":[adf[1]],
            "Coin1 1% Conf":[adf[4]['1%']],
            "Coin1 5% Conf":[adf[4]['5%']],
            "Coin1 10% Conf":[adf[4]['10%']],
            "Coin2 ADF Stat":[adf2[0]],
            "Coin2 p-value":[adf2[1]],
            "Coin2 1% Conf":[adf2[4]['1%']],
            "Coin2 5% Conf":[adf2[4]['5%']],
            "Coin2 10% Conf":[adf2[4]['10%']],
            "Coin1 Max":[np.max(coin1["Price"])],
            "Coin1 Min":[np.min(coin1["Price"])],
            "Coin1 Mean":[np.mean(coin1["Price"])],
            "Coin1 Std Dev":[np.std(coin1["Price"])],
            "Coin2 Max":[np.max(coin2["Price"])],
            "Coin2 Min":[np.min(coin2["Price"])],
            "Coin2 Mean":[np.mean(coin2["Price"])],
            "Coin2 Std Dev":[np.std(coin2["Price"])]
        }
        
    def runPairsTradingStrategy(self,coin1,coin2,sym1,sym2):
        
        if DEBUG: print("Running pair ",sym1,"/",sym2,"...",sep="")
        cash=startCash=self.startCash #equally weighted
        #profit=0.0 #value weighted
        hilong=hishort=lolong=loshort=hi=lo=firstlo=firsthi=0.0
        i=0
        lags=self.lags
        tight=self.tight
        wide=self.wide
        notTight=notWide=True
        for prc in coin1["Price"]:
            if i>lags:
                #spread
                coin1mn = np.mean(coin1["Price"][i-lags:i])
                coin2mn = np.mean(coin2["Price"][i-lags:i])
                if coin2mn > coin1mn:
                    himn,lomn = coin2mn,coin1mn 
                else: 
                    himn,lomn = coin1mn,coin2mn#calcing spread
                spdmn = himn-lomn
                
                #current
                coin1curr = coin1["Price"][i]
                coin2curr = coin2["Price"][i]
                date = coin1["Date"][i]
                if coin2curr>coin1curr:
                    hi,lo = coin2curr,coin1curr
                else: 
                    hi,lo = coin1curr,coin2curr
                spd = hi-lo
                
                if spd < spdmn*tight and notTight:
                    hilong=hi
                    loshort=lo
                    if lolong != 0.0 and loshort != 0.0: 
                        if DEBUG: print(date," tight, shorting lo price ",lo," and longing hi price ",hi)#," a trade profit of: ",loshort-lolong,end=' ')
                        if DEBUG: print("hishort,hilong: ",hishort,hilong,hishort/hilong)
                        if DEBUG: print("loshort,lolong: ",loshort,lolong,loshort/lolong)
                        if DEBUG: print("cash before: ",cash)
                        if DEBUG: print("hi cash before/after: ",cash*0.5,cash*0.5*(hishort/hilong))
                        if DEBUG: print("lo cash before/after: ",cash*0.5,cash*0.5*(loshort/lolong))
                        cash=cash*0.5*(loshort/lolong) + cash*0.5*(hishort/hilong)
                        if DEBUG: print("cash after: ",cash)
                        
                        #loshort=lolong=0.0
                    if firsthi == 0.0: firsthi=hi
                    notTight=False
                    notWide=True
                elif spd > spdmn*wide and notWide:
                    hishort=hi
                    lolong=lo
                    if hilong != 0.0 and hishort != 0.0: 
                        if DEBUG: print(date," wide, shorting hi price ",hi," and longing lo price ",lo)#," a trade profit of: ",hishort-hilong,end=' ')
                        if DEBUG: print("hishort,hilong: ",hishort,hilong,hishort/hilong)
                        if DEBUG: print("loshort,lolong: ",loshort,lolong,loshort/lolong)
                        if DEBUG: print("cash before: ",cash)
                        if DEBUG: print("hi cash: ",cash*0.5*(hishort/hilong))
                        if DEBUG: print("lo cash: ",cash*0.5*(loshort/lolong))
                        cash=cash*0.5*(loshort/lolong) + cash*0.5*(hishort/hilong)
                        if DEBUG: print("cash after: ",cash)
                        #hishort=hilong=0.0
                    if firstlo == 0.0: firstlo=lo
                    notWide=False
                    notTight=True
                else:
                    pass
            i+=1
        if DEBUG: print("starting cash: ",startCash)
        if DEBUG: print("final cash: ",frmt2(cash))
        if DEBUG: print("returns: ",frmt2(((cash-startCash)/startCash)*100),"%\n")
        
        return {"returns":[((cash-startCash)/startCash)*100]}
    
def frmt4(num): return"{0:.4f}".format(num)
def frmt2(num): return"{0:.2f}".format(num)

def main():
    trdr1 = cryptoPairsTrader()
    trdr1.loadData()
    trdr1.runAllPairs()
    
main()