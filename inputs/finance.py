from inputs import InputStream
import config
import os
from utils.sparse_vector import SparseVector

import math
import time
import collections as cl
import numpy as np
import random as rd
import random


subset_for_large_collection = ["AMEX:BCV","AMEX:ESP","AMEX:SEB","CPH:NOVO+B","CPH:NZYM+B","NASDAQ:ACET","NASDAQ:ADP","NASDAQ:ALCO","NASDAQ:ALOG","NASDAQ:AMAT","NASDAQ:ANAT","NASDAQ:ARDNA","NASDAQ:ASBC","NASDAQ:ASEI","NASDAQ:ATRI","NASDAQ:BOBE","NASDAQ:BSET","NASDAQ:CBSH","NASDAQ:CINF","NASDAQ:CMCSA","NASDAQ:COHR","NASDAQ:COKE","NASDAQ:CTWS","NASDAQ:DFZ","NASDAQ:DIOD","NASDAQ:EDUC","NASDAQ:ERIC","NASDAQ:ESCA","NASDAQ:FITB","NASDAQ:FLXS","NASDAQ:GKSR","NASDAQ:GLDC","NASDAQ:HAS","NASDAQ:HBAN","NASDAQ:HELE","NASDAQ:HWKN","NASDAQ:INTC","NASDAQ:KAMN","NASDAQ:KBALB","NASDAQ:KCLI","NASDAQ:KELYA","NASDAQ:KEQU","NASDAQ:KLIC","NASDAQ:LANC","NASDAQ:LAWS","NASDAQ:MGEE","NASDAQ:MKTAY","NASDAQ:MLHR","NASDAQ:MOLX","NASDAQ:MSEX","NASDAQ:MSW","NASDAQ:MTSC","NASDAQ:MYL","NASDAQ:NTRS","NASDAQ:NWLI","NASDAQ:OTTR","NASDAQ:PCAR","NASDAQ:PLFE","NASDAQ:RAVN","NASDAQ:SGC","NASDAQ:SHLM","NASDAQ:SIAL","NASDAQ:SIGI","NASDAQ:SMTC","NASDAQ:SYNL","NASDAQ:TRMK","NASDAQ:TWIN","NASDAQ:TXN","NASDAQ:UMBF","NASDAQ:VLGEA","NASDAQ:WABC","NASDAQ:WEYS","NASDAQ:WSCI","NYSE:AA","NYSE:ABM","NYSE:ABT","NYSE:ADI","NYSE:ADM","NYSE:ADX","NYSE:AEP","NYSE:AET","NYSE:AFL","NYSE:AGL","NYSE:AIT","NYSE:ALE","NYSE:ALX","NYSE:AM","NYSE:AMD","NYSE:AME","NYSE:APA","NYSE:APD","NYSE:ASA","NYSE:ASH","NYSE:AVA","NYSE:AVP","NYSE:AWR","NYSE:AXP","NYSE:AZZ","NYSE:B","NYSE:BA"]#,"NYSE:BAC    ","NYSE:BAX","NYSE:BBT","NYSE:BCE","NYSE:BCR","NYSE:BDX","NYSE:BF.B","NYSE:BGG","NYSE:BKH","NYSE:BLL","NYSE:BMI","NYSE:BMS","NYSE:BMY","NYSE:BOH","NYSE:BP","NYSE:BRE","NYSE:BRS","NYSE:BUD","NYSE:CAE","NYSE:CAG","NYSE:CAJ","NYSE:CAT","NYSE:CB","NYSE:CBE","NYSE:CBT","NYSE:CEG","NYSE:CFR","NYSE:CHD","NYSE:CHE","NYSE:CHG","NYSE:CL","NYSE:CLC","NYSE:CLF","NYSE:CLX","NYSE:CMI","NYSE:CMS","NYSE:CNA","NYSE:CNP","NYSE:CNW","NYSE:COP","NYSE:CPB","NYSE:CPK","NYSE:CRD.B","NYSE:CRS","NYSE:CSC","NYSE:CSL","NYSE:CTS","NYSE:CUB","NYSE:CV","NYSE:CVS","NYSE:CWT","NYSE:CYN","NYSE:D","NYSE:DCI","NYSE:DD","NYSE:DDS","NYSE:DE","NYSE:DIS","NYSE:DLX","NYSE:DOV","NYSE:DOW","NYSE:DPL","NYSE:DTE","NYSE:DUK","NYSE:EBF","NYSE:ECL","NYSE:ED","NYSE:EDE","NYSE:EFX","NYSE:EGN","NYSE:EGP","NYSE:EIX","NYSE:EMR","NYSE:EQT","NYSE:ETN","NYSE:ETR","NYSE:EV","NYSE:EXC","NYSE:F","NYSE:FAF","NYSE:FDI","NYSE:FDO","NYSE:FHN","NYSE:FL","NYSE:FRT","NYSE:FRX","NYSE:FSS","NYSE:FST","NYSE:FUL","NYSE:GAS","NYSE:GCI","NYSE:GCO","NYSE:GD","NYSE:GE","NYSE:GFI","NYSE:GGG","NYSE:GIS","NYSE:GLT","NYSE:GLW","NYSE:GMT","NYSE:GNI","NYSE:GPC","NYSE:GPS","NYSE:GR","NYSE:GSK","NYSE:GWW","NYSE:GXP","NYSE:HAL","NYSE:HAS","NYSE:HE","NYSE:HEI","NYSE:HES","NYSE:HIT","NYSE:HL","NYSE:HMC","NYSE:HNI","NYSE:HNZ","NYSE:HP","NYSE:HPQ","NYSE:HRB","NYSE:HRC","NYSE:HRL","NYSE:HSC","NYSE:HSY","NYSE:HUB.B","NYSE:HXL","NYSE:IBM","NYSE:ICB","NYSE:IDA","NYSE:IFF","NYSE:IP","NYSE:IPG","NYSE:IR","NYSE:ITW","NYSE:JCI","NYSE:JCP","NYSE:JNJ","NYSE:JPM","NYSE:JW.A","NYSE:JWN","NYSE:K","NYSE:KEY","NYSE:KMB","NYSE:KMT","NYSE:KO","NYSE:KUB","NYSE:KWR","NYSE:L","NYSE:LEG","NYSE:LEN","NYSE:LG","NYSE:LLY","NYSE:LNC","NYSE:LNT","NYSE:LOW","NYSE:LPX","NYSE:LTD","NYSE:LUK","NYSE:LZB","NYSE:MCD","NYSE:MCS","NYSE:MDP","NYSE:MDT","NYSE:MDU","NYSE:MEE","NYSE:MHP","NYSE:MKC","NYSE:MMC","NYSE:MMM","NYSE:MO","NYSE:MOT","NYSE:MPR","NYSE:MRK","NYSE:MSA","NYSE:MSI","NYSE:MTB","NYSE:MTZ","NYSE:MUR","NYSE:MWV","NYSE:NAV","NYSE:NBL","NYSE:NC","NYSE:NEM","NYSE:NEU","NYSE:NFG","NYSE:NI","NYSE:NJR","NYSE:NOC","NYSE:NPK","NYSE:NRT","NYSE:NU","NYSE:NUE","NYSE:NWE","NYSE:NWL","NYSE:NWN","NYSE:NXY","NYSE:OGE","NYSE:OII","NYSE:OKE","NYSE:OLN"]

class DriftStockPrices(InputStream):
    m_stDataset = "SP100_2007_2011"
    def __init__(self, nodes, driftProb = 0.028, prediction_index = 1, numberOfStocks = None, normalize = False):
        self.m_lFeatures    = []
        self.m_aDates       = []
        self.m_aStocks      = []
        self.m_lAvgPrices11 = cl.defaultdict(list)
        self.m_lAvgPrices50 = cl.defaultdict(list)
        self.m_lAvgPrices200= cl.defaultdict(list)
        self.m_hPrices      = cl.defaultdict(list)
        self.bNormalize = normalize
        self.drift_prob = driftProb
        self.prediction_index = prediction_index
        self.configureFiles()
        self.readPrices(numberOfStocks)
        self.preprocessFeatures()
        self.m_lAvgPrices11.clear()
        self.m_lAvgPrices50.clear()
        self.m_lAvgPrices200.clear()
        self.m_stActStock = self.m_hPrices.keys()[0]
        InputStream.__init__(self, self.getIdentifier, nodes)

    def getIdentifier(self):
        return "DriftingStockPrices"
    
    def has_more_examples(self):
        return True
    
    def _generate_example(self):
        iNumberOfFeaturesPerStock = 4
        iPosCurStock = self.m_aStocks.index(self.m_stActStock)
        iTargetStockPos = iNumberOfFeaturesPerStock*iPosCurStock
        iFeatureIndex = self.prediction_index - 1
        if iFeatureIndex < 0:
            iFeatureIndex = len(self.m_lFeatures)-1                
        record      = SparseVector()
        features    = self.getFeatures(iFeatureIndex)
        label       = self.getLabel(self.prediction_index)
        for i in xrange(len(features)):
            val = features[i]
            if math.isnan(val) or math.isinf(val):
                print "Error, invalid feature value: "+str(val)
                val = 0.0
            #the following code ensures that the features of the target stock are always at the beginning of the feature vector
#             if i < iTargetStockPos or i >= iTargetStockPos + iNumberOfFeaturesPerStock:
#                 record[i+iNumberOfFeaturesPerStock] = val
#             else:
#                 record[i-iTargetStockPos] = val
            record[i] = val
        self.prediction_index += 1
        if self.prediction_index >= len(self.m_hPrices[self.m_stActStock]):
            self.prediction_index = 0
        if self.generated_examples % self.number_of_nodes==0:
            self.drift()
        return (record,label)
    
    def drift(self):
        if random.random() < self.drift_prob:
            iNewStock = random.randint(0,len(self.m_aStocks)-1)
            self.m_stActStock = self.m_aStocks[iNewStock]
    
    def getFeatures(self, t):
        return self.m_lFeatures[t]
    
    def getLabel(self, t):
        return self.m_hPrices[self.m_stActStock][t]
        
    def configureFiles(self):
        self.m_stPrices = os.path.join(config.PATH_TO_FINANCE, self.m_stDataset+".fdc")
    
    def preprocessAvgPrices(self):
        print "Preprocessing average prices:",
        for stock in self.m_hPrices.keys():
            print stock,
            for t in xrange(len(self.m_hPrices[stock])):
                if not math.isnan(self.m_hPrices[stock][t]):
                    self.m_lAvgPrices11[stock].append(self.getAvgPrice(stock, t, 11))
                    self.m_lAvgPrices50[stock].append(self.getAvgPrice(stock, t, 50))
                    self.m_lAvgPrices200[stock].append(self.getAvgPrice(stock, t, 200))
                else:
                    self.m_lAvgPrices11[stock].append(0.0)
                    self.m_lAvgPrices50[stock].append(0.0)
                    self.m_lAvgPrices200[stock].append(0.0)
        print " done."
    
    def preprocessFeatures(self):
        self.preprocessAvgPrices()
        print "Preprocessing features:"
        for t in xrange(len(self.m_hPrices[self.m_hPrices.keys()[0]])):                
            if t>0: #t is the predicted day, its features are from t-1, thus, t-1 should be greater or equal 0
                #try:
                features = self.getPreprocessedFeatures(t-1)
                self.m_lFeatures.append(features)
                #except:
                #    print t
                #    print val                
        print "Preprocessing done."
    
    def getPreprocessedFeatures(self, t):
        features = []    
        for stock in self.m_hPrices.keys():
            if not math.isnan(self.m_hPrices[stock][t]):             
                features.append(self.m_hPrices[stock][t])
                features.append(self.m_lAvgPrices11[stock][t])
                features.append(self.m_lAvgPrices50[stock][t])
                features.append(self.m_lAvgPrices200[stock][t])
            else:
                features.append(0.0)
                features.append(0.0)
                features.append(0.0)
                features.append(0.0)
        return features
    
    def getAvgPrice(self, stock, t, window):
        if t+1-window<0:
            window = t+1
        vals = np.array([val for val in self.m_hPrices[stock][t+1-window:t+1] if not math.isnan(val)])
        return np.mean(vals)
    
    def getSubsampleOfStocks(self, aStocks, numberOfStocks):
        if numberOfStocks == 300: #cheap hack
            return subset_for_large_collection
        return [ aStocks[i] for i in rd.sample(xrange(len(aStocks)), numberOfStocks) ]
#         return [ aStocks[i] for i in xrange(numberOfStocks) ]
    
    def readPrices(self, numberOfStocks = None):
        print "Reading prices:",
        if len(self.m_hPrices.keys()) > 0:
            return
        
        self.m_hPrices.clear()
        f = open(self.m_stPrices, 'r')
        aCollection = f.readlines()
        f.close()
        aStocks = aCollection[0].split()
        aStocks = aStocks[1:len(aStocks)]
        
        self.m_aDates[:] = []
        if numberOfStocks is not None:
            self.m_aStocks = self.getSubsampleOfStocks(aStocks, numberOfStocks)
        else:
            self.m_aStocks = aStocks
        iCount = 0
        iValCount = 0
        iNanValCount = 0
        for line in aCollection[1:len(aCollection)]:
            if iCount % 100 == 0:
                print ".",
            iCount += 1
            aLine = line.split()
            self.m_aDates.append(aLine[0])
            for i in xrange(1, len(aLine)):
                val = float(aLine[i])                
                if numberOfStocks is not None:
                    stock = aStocks[i-1]
                    if stock in self.m_aStocks:
                        if math.isnan(val):
                            iNanValCount +=1
                            val = 0.0
                        iValCount += 1
                        self.m_hPrices[stock].append(val)
                else:
                    if math.isnan(val):
                        iNanValCount +=1
                        val = 0.0
                    iValCount += 1
                    self.m_hPrices[aStocks[i-1]].append(val)
        if iNanValCount > 0:
            print "Price-values that were NaN: "+str(iNanValCount) + " of " + str(iValCount)
        print "done."
        if self.bNormalize:
            print "Normalizing prices...",
            for stock in self.m_aStocks:
                price = np.array(self.m_hPrices[stock])
                avg = np.average(price)
                std = math.sqrt(np.var(price))
                for i in xrange(len(self.m_hPrices[stock])):
                    self.m_hPrices[stock][i] = (self.m_hPrices[stock][i] - avg) / std 
            print "done."
    
class StockPrices(InputStream):    
    m_stDataset = "large_collection"
    
    def __init__(self, nodes, prediction_index = 1, bPreprocess = False, numberOfStocks = None):
        self.m_lFeatures    = []
        self.m_lLabels      = []
        self.m_aDates       = []
        self.m_aStocks      = []
        self.m_lAvgPrices11 = cl.defaultdict(list)
        self.m_lAvgPrices50 = cl.defaultdict(list)
        self.m_lAvgPrices200= cl.defaultdict(list)
        self.m_hPrices      = cl.defaultdict(list)
        self.m_bPreprocess  = bPreprocess
        self.prediction_index = prediction_index
        self.configureFiles()
        self.readPrices(numberOfStocks)
        if self.m_bPreprocess:
            self.preprocessFeatures()
            self.m_hPrices.clear() #clear off some memory
            self.m_lAvgPrices11.clear()
            self.m_lAvgPrices50.clear()
            self.m_lAvgPrices200.clear()
        else:
            self.m_stActStock = self.m_hPrices.keys()[0]
        
        InputStream.__init__(self, self.getIdentifier, nodes)
    
    def getIdentifier(self):
        return "StockPrices"
    
    def has_more_examples(self):
        if self.m_bPreprocess:
            return self.prediction_index < len(self.m_lFeatures)
        else:
            if self.m_stActStock != self.m_hPrices.keys()[-1]:
                return True
            else:
                if self.prediction_index >= len(self.m_hPrices[self.m_stActStock]):
                    return False
                else:
                    return True
    
    def _generate_example(self):
        record      = SparseVector()        
        features    = self.getFeatures(self.prediction_index-1)
        label       = self.getLabel(self.prediction_index)
        if math.isnan(label) or math.isinf(label):
            print "Error, invalid label: "+str(label)
            label = 0.0
        for i,val in enumerate(features):
            if math.isnan(val) or math.isinf(val):
                print "Error, invalid feature value: "+str(val)
                val = 0.0
            record[i] = val
        self.prediction_index += 1
        return (record,label)
    
    def getFeatures(self, t):
        if self.m_bPreprocess:
            return self.m_lFeatures[t]
        t = self.getNextValidTime(t)
        if math.isnan(self.m_hPrices[self.m_stActStock][t]):
            print "oops"
        features = [self.m_hPrices[self.m_stActStock][t]]                       #price of this stock
        features.append(self.getAvgPrice(self.m_stActStock,t,11))                 #avg price of  11 days of this stock
        features.append(self.getAvgPrice(self.m_stActStock,t,50))                 #avg price of  50 days of this stock
        features.append(self.getAvgPrice(self.m_stActStock,t,200))                #avg price of 200 days of this stock
        for other_stock in self.m_hPrices.keys():
            if other_stock != self.m_stActStock:
                if not math.isnan(self.m_hPrices[other_stock][t]):
                    features.append(self.m_hPrices[other_stock][t])         #prices of all other stocks
                    features.append(self.getAvgPrice(other_stock,t,11))   #avg prices of  11 days of all other stock
                    features.append(self.getAvgPrice(other_stock,t,50))   #avg prices of  50 days of all other stock
                    features.append(self.getAvgPrice(other_stock,t,200))   #avg prices of 200 days of all other stock
        self.prediction_index = t+1
        return features
            
    def getNextValidTime(self, t):
        if t >= len(self.m_hPrices[self.m_stActStock]):
            idxNextStock = self.m_hPrices.keys().index(self.m_stActStock)+1
            if idxNextStock >= len(self.m_hPrices.keys()):
                return float('inf')
            self.m_stActStock = self.m_hPrices.keys()[idxNextStock]
            self.prediction_index = 0
            t=0
        if math.isnan(self.m_hPrices[self.m_stActStock][t]):
            for i in xrange(t,len(self.m_hPrices[self.m_stActStock])):
                if not math.isnan(self.m_hPrices[self.m_stActStock][i]):
                    return i
            idxNextStock = self.m_hPrices.keys().index(self.m_stActStock)+1
            if idxNextStock >= len(self.m_hPrices.keys()):
                return float('inf')
            self.m_stActStock = self.m_hPrices.keys()[idxNextStock]
            self.prediction_index = 0
            return self.getNextValidTime(t)
        else:
            return t
        
    def getLabel(self, t):
        if self.m_bPreprocess:
            return self.m_lLabels[t]
        else:
            t = self.getNextValidTime(t)
            return self.m_hPrices[self.m_stActStock][t]
        
    def configureFiles(self):
        self.m_stPrices         = os.path.join(config.PATH_TO_FINANCE, self.m_stDataset+".fdc")
    
    def preprocessAvgPrices(self):
        print "Preprocessing average prices:",
        for stock in self.m_hPrices.keys():
            print stock,
            for t in xrange(len(self.m_hPrices[stock])):
                if not math.isnan(self.m_hPrices[stock][t]):
                    self.m_lAvgPrices11[stock].append(self.getAvgPrice(stock, t, 11))
                    self.m_lAvgPrices50[stock].append(self.getAvgPrice(stock, t, 50))
                    self.m_lAvgPrices200[stock].append(self.getAvgPrice(stock, t, 200))
                else:
                    self.m_lAvgPrices11[stock].append(0.0)
                    self.m_lAvgPrices50[stock].append(0.0)
                    self.m_lAvgPrices200[stock].append(0.0)
        print " done."
    
    def preprocessFeatures(self):
        self.preprocessAvgPrices()
        print "Preprocessing features:"
        for stock in self.m_hPrices.keys():
            print stock+":",
            for t,val in enumerate(self.m_hPrices[stock]):                
                if not math.isnan(val) and t>0: #t is the predicted day, its features are from t-1, thus, t-1 should be greater or equal 0
                    #try:
                    features = self.getPreprocessedFeatures(stock, t-1)
                    self.m_lFeatures.append(features)
                    self.m_lLabels.append(val)
                    #except:
                    #    print t
                    #    print val                
            print "done"
        print "Preprocessing done."
    
    def getPreprocessedFeatures(self, stock, t):
        features = [self.m_hPrices[stock][t]]                       #price of this stock
        features.append(self.m_lAvgPrices11[stock][t])                 #avg price of  11 days of this stock
        features.append(self.m_lAvgPrices50[stock][t])                 #avg price of  50 days of this stock
        features.append(self.m_lAvgPrices200[stock][t])                #avg price of 200 days of this stock
        for other_stock in self.m_hPrices.keys():
            if other_stock != stock:
                if not math.isnan(self.m_hPrices[other_stock][t]):                    
                    features.append(self.m_hPrices[other_stock][t])         #prices of all other stocks
                    features.append(self.m_lAvgPrices11[other_stock][t])   #avg prices of  11 days of all other stock
                    features.append(self.m_lAvgPrices50[other_stock][t])   #avg prices of  50 days of all other stock
                    features.append(self.m_lAvgPrices200[other_stock][t])  #avg prices of 200 days of all other stock
                else:
                    features.append(0.0)
                    features.append(0.0)
                    features.append(0.0)
                    features.append(0.0)
        return features
                    
                
    def getAvgPrice(self, stock, t, window):
        if t+1-window<0:
            window = t+1
        vals = np.array([val for val in self.m_hPrices[stock][t+1-window:t+1] if not math.isnan(val)])
        return np.mean(vals)
    
    def getSubsampleOfStocks(self, aStocks, numberOfStocks):
        if numberOfStocks == 300: #cheap hack
            return subset_for_large_collection
        return [ aStocks[i] for i in rd.sample(xrange(len(aStocks)), numberOfStocks) ]
    
    def readPrices(self, numberOfStocks = None):
        print "Reading prices:",
        if len(self.m_hPrices.keys()) > 0:
            return
        
        self.m_hPrices.clear()
        f = open(self.m_stPrices, 'r')
        aCollection = f.readlines()
        f.close()
        aStocks = aCollection[0].split()
        aStocks = aStocks[1:len(aStocks)]
        
        self.m_aDates[:] = []
        if numberOfStocks is not None:
            self.m_aStocks = self.getSubsampleOfStocks(aStocks, numberOfStocks)
        else:
            self.m_aStocks = aStocks
        iCount = 0
        iValCount = 0
        iNanValCount = 0
        for line in aCollection[1:len(aCollection)]:
#             if iCount % 100 == 0:
#                 print ".",
            iCount += 1
            aLine = line.split()
            self.m_aDates.append(aLine[0])
            for i in xrange(1, len(aLine)):
                val = float(aLine[i])                
                if numberOfStocks is not None:
                    stock = aStocks[i-1]
                    if stock in self.m_aStocks:
                        if math.isnan(val):
                            iNanValCount +=1
                            val = 0.0
                        iValCount += 1
                        self.m_hPrices[stock].append(val)
                else:
                    if math.isnan(val):
                        iNanValCount +=1
                        val = 0.0
                    iValCount += 1
                    self.m_hPrices[aStocks[i-1]].append(val)
        if iNanValCount > 0:
            print "Price-values that were NaN: "+str(iNanValCount) + " of " + str(iValCount)
        print "done."
    
#Possible Stocks as targets:
#SP100: [["NASDAQ:AAPL"],[ "NASDAQ:AMGN"],[ "NASDAQ:AMZN"],[ "NASDAQ:CMCSA"],[ "NASDAQ:COST"],[ "NASDAQ:CSCO"],[ "NASDAQ:DELL"],[ "NASDAQ:GILD"],[ "NASDAQ:GOOG"],[ "NASDAQ:INTC"],[ "NASDAQ:MSFT"],[ "NASDAQ:NWSA"],[ "NASDAQ:ORCL"],[ "NASDAQ:QCOM"],[ "NYSE:AA"],[ "NYSE:ABT"],[ "NYSE:AEP"],[ "NYSE:ALL"],[ "NYSE:AVP"],[ "NYSE:AXP"],[ "NYSE:BA"],[ "NYSE:BAC"],[ "NYSE:BAX"],[ "NYSE:BHI"],[ "NYSE:BMY"],[ "NYSE:C"],[ "NYSE:CAT"],[ "NYSE:CL"],[ "NYSE:COF"],[ "NYSE:COP"],[ "NYSE:CPB"],[ "NYSE:CVS"],[ "NYSE:CVX"],[ "NYSE:DD"],[ "NYSE:DIS"],[ "NYSE:DOW"],[ "NYSE:DVN"],[ "NYSE:EMC"],[ "NYSE:ETR"],[ "NYSE:EXC"],[ "NYSE:F"],[ "NYSE:FCX"],[ "NYSE:FDX"],[ "NYSE:GD"],[ "NYSE:GE"],[ "NYSE:HAL"],[ "NYSE:HD"],[ "NYSE:HNZ"],[ "NYSE:HON"],[ "NYSE:HPQ"],[ "NYSE:IBM"],[ "NYSE:JNJ"],[ "NYSE:JPM"],[ "NYSE:KFT"],[ "NYSE:KO"],[ "NYSE:LMT"],[ "NYSE:LOW"],[ "NYSE:MA"],[ "NYSE:MCD"],[ "NYSE:MDT"],[ "NYSE:MET"],[ "NYSE:MMM"],[ "NYSE:MO"],[ "NYSE:MON"],[ "NYSE:MRK"],[ "NYSE:MS"],[ "NYSE:NKE"],[ "NYSE:NOV"],[ "NYSE:NSC"],[ "NYSE:NYX"],[ "NYSE:OXY"],[ "NYSE:PEP"],[ "NYSE:PFE"],[ "NYSE:PG"],[ "NYSE:RTN"],[ "NYSE:SLB"],[ "NYSE:SLE"],[ "NYSE:SO"],[ "NYSE:T"],[ "NYSE:TGT"],[ "NYSE:TWX"],[ "NYSE:UNH"],[ "NYSE:UPS"],[ "NYSE:USB"],[ "NYSE:UTX"],[ "NYSE:VZ"],[ "NYSE:WFC"],[ "NYSE:WMB"],[ "NYSE:WMT"],[ "NYSE:WY"],[ "NYSE:XOM"],[ "NYSE:XRX"]]
#DAX30: [["ETR:ADS", "ETR:ALV", "ETR:BAS", "ETR:BEI", "ETR:BMW", "ETR:CBK", "ETR:DAI", "ETR:DB1", "ETR:DBK", "ETR:DPW", "ETR:DTE", "ETR:EOAN", "ETR:FME", "ETR:FRE", "ETR:HEI", "ETR:HEN3", "ETR:IFX", "ETR:LHA", "ETR:LIN", "ETR:MEO", "ETR:MRK", "ETR:MUV2", "ETR:RWE", "ETR:SAP", "ETR:SDF", "ETR:SIE", "ETR:TKA", "ETR:VOW3"]]

class StockPriceFeatureStream(InputStream):
    SP100 = 'SP100_2007_2011'
    DAX30 = 'DAX30_2007_2011'
    EPS   = 'annotation_%s_eps'
    FV    = 'annotation_%s_fv'
    PRICE = 'annotation_%s_price'
    
    m_aPrices       = cl.defaultdict(list)
    m_aAnnotations  = cl.defaultdict(list)
    m_aAvgPrices11  = cl.defaultdict(list)
    m_aAvgPrices50  = cl.defaultdict(list)
    m_aAvgPrices200 = cl.defaultdict(list)
    m_aFeatureNames = []
    m_aDates        = []
    m_aTargetStocks = []
    m_iAccountingPeriod = 62
    
    def __init__(self, identifier, nodes, dataset, label, targetStock = 'NASDAQ:GOOG', repetitions = 0, repetition_interval = 7, delay = False, prediction_index = 0):
        self.prediction_index = prediction_index
        if isinstance(targetStock, list):
            self.m_stTargetStock = targetStock.pop(0)
            self.m_aTargetStocks = targetStock
        else:
            self.m_stTargetStock = targetStock
        self.configureFiles(dataset, label)
        self.readPrices()
        self.readAnnotations()
        self.preprocessAvgPrices()        
        self.m_oCorrel = correlation(self.m_aPrices, [self.m_stTargetStock])
        self.calcFeatureNames()
        self.parameters = "Financial(%s)" % dataset
        self.m_bDelay = False
        self.m_iRepetitions = repetitions
        self.m_iRepetitionInterval = repetition_interval
        self.m_iCurrRepitions = 0
        InputStream.__init__(self, identifier, nodes)
        
    def has_more_examples(self):
        return self.prediction_index < len(self.m_aPrices[self.m_stTargetStock])    
    
    def generate_example(self):
        record = SparseVector()
        for i in xrange(len(self.m_aFeatureNames)):
            aFeatures = self.getFeatures(self.prediction_index)
            record.components[self.m_aFeatureNames[i]] = aFeatures[i]        
        label = self.m_aAnnotations[self.m_stTargetStock][self.prediction_index]
        if self.m_bDelay == True:
            if self.prediction_index % self.m_iAccountingPeriod == 0:
                label = []
                for t in xrange(self.prediction_index - self.m_iAccountingPeriod + 1, self.prediction_index + 1):
                    if t >= 0:
                        label.append(self.m_aAnnotations[self.m_stTargetStock][t])
            else:
                label = None
                
        self.prediction_index += 1
        if self.prediction_index % self.m_iRepetitionInterval == 0:
            if self.m_iCurrRepitions < self.m_iRepetitions:
                self.prediction_index -= self.m_iRepetitionInterval
                self.m_iCurrRepitions += 1
            else:
                self.m_iCurrRepitions = 0
        if self.prediction_index >= len(self.m_aPrices[self.m_stTargetStock]):
            if len(self.m_aTargetStocks) > 0:
                self.m_stTargetStock = self.m_aTargetStocks.pop(0)
                self.m_oCorrel.changeSet([self.m_stTargetStock])
                self.prediction_index = 0
        
        return (record, label)
        
    def configureFiles(self, dataset, label):
        print dataset, label
        self.m_stPrices         = os.path.join(config.PATH_TO_FINANCE, dataset+".fdc")                      
#         self.m_stAnnotations    = os.path.join(config.PATH_TO_FINANCE, (label%(dataset))+".daf")
        self.m_stAnnotations    = os.path.join(config.PATH_TO_FINANCE, (label)+".daf")
            
    def readPrices(self):
        if len(self.m_aPrices.keys()) > 0:
            return
        
        self.m_aPrices.clear()
        f = open(self.m_stPrices, 'r')
        aCollection = f.readlines()
        f.close()
        aStocks = aCollection[0].split()
        aStocks = aStocks[1:len(aStocks)]
        self.m_aStocks = aStocks
        self.m_aDates[:] = []
        
        for line in aCollection[1:len(aCollection)]:
            aLine = line.split()
            self.m_aDates.append(aLine[0])
            for i in xrange(1, len(aLine)):
                self.m_aPrices[aStocks[i-1]].append(float(aLine[i]))
                
    def readAnnotations(self):
        f = open(self.m_stAnnotations, 'r')
        aCollection = f.readlines()
        f.close()
        aStocks = aCollection[0].split()
        aStocks = aStocks[1:len(aStocks)]
        self.m_aDates[:] = []
        self.m_aAnnotations.clear()
        for line in aCollection[1:len(aCollection)]:
            aLine = line.split()
            self.m_aDates.append(aLine[0])
            for i in range(1, len(aLine)):
                self.m_aAnnotations[aStocks[i-1]].append(float(aLine[i]))
    
    def preprocessAvgPrices(self):
        print "Preprocess Avg. Prices: "        
        self.calcAvgPrices(self.m_aAvgPrices11,  11)        
        self.calcAvgPrices(self.m_aAvgPrices50,  50)
        self.calcAvgPrices(self.m_aAvgPrices200, 200)
        print "Preprocessing done."
            
    def getAvgPrices(self, t, window):
        aAvgPrices = None
        if window == 11:
            aAvgPrices = self.m_aAvgPrices11
        elif window == 50:
            aAvgPrices = self.m_aAvgPrices50
        elif window == 200:
            aAvgPrices = self.m_aAvgPrices200
        else:
            print "Invalid price window: "+str(window)
            return []
        aPrices = []
        for stock in self.m_oDataIO.m_aPrices.keys():
            aPrices += [aAvgPrices[stock][t]]
        return aPrices
    
    def calcAvgPrices(self, aAvgPrices, window):
        for t in xrange(len(self.m_aDates)):
            actWindow = window
#             print ".",
            if t - actWindow < 0:
                actWindow = t
            for stock in self.m_aPrices.keys():
                dVal = 0.0
                for i in xrange(t-actWindow, t):
                    dVal += self.m_aPrices[stock][i]
                dCount = float(actWindow)
                if dCount == 0.0:
                    dCount = 1.0
                aAvgPrices[stock] += [dVal / dCount]
        print "avg calculated..."
        print "\n"
        
    def calcFeatureNames(self):
        self.m_aFeatureNames  = ["P(%s)" % self.m_stTargetStock]
        self.m_aFeatureNames += ["AvgP(%s,11)" % self.m_stTargetStock]
        self.m_aFeatureNames += ["AvgP(%s,50)" % self.m_stTargetStock]
        self.m_aFeatureNames += ["AvgP(%s,200)" % self.m_stTargetStock]
        for (x,y) in self.m_oCorrel.m_lPairsRelated:
            if (x == self.m_stTargetStock):
                self.m_aFeatureNames += ["AvgP(%s,11)*c_11(%s,%s)" % (y,x,y)]
            else:
                self.m_aFeatureNames += ["AvgP(%s,11)*c_11(%s,%s)" % (x,y,x)]
        for (x,y) in self.m_oCorrel.m_lPairsRelated:
            if (x == self.m_stTargetStock):
                self.m_aFeatureNames += ["AvgP(%s,50)*c_50(%s,%s)" % (y,x,y)]
            else:
                self.m_aFeatureNames += ["AvgP(%s,50)*c_11(%s,%s)" % (x,y,x)]
        for (x,y) in self.m_oCorrel.m_lPairsRelated:
            if (x == self.m_stTargetStock):
                self.m_aFeatureNames += ["AvgP(%s,200)*c_200(%s,%s)" % (y,x,y)]
            else:
                self.m_aFeatureNames += ["AvgP(%s,200)*c_11(%s,%s)" % (x,y,x)]
    
    def getFeatures(self, t):
        features  = [self.m_aPrices[self.m_stTargetStock][t]]
        features.append(self.m_aAvgPrices11[self.m_stTargetStock][t])
        features.append(self.m_aAvgPrices50[self.m_stTargetStock][t])
        features.append(self.m_aAvgPrices200[self.m_stTargetStock][t])
        features += self.m_oCorrel.getWeightedCorrelsRelated(t,11)
        features += self.m_oCorrel.getWeightedCorrelsRelated(t,50)
        features += self.m_oCorrel.getWeightedCorrelsRelated(t,200)
        return features
    
    def getIdentifier(self):
        return self.parameters   
        

class correlation:
    m_lPairs            = []
    m_lPairsRelated     = []
    m_aTimeSeries       = None
    m_lSet              = []
    m_lCorrel11         = cl.defaultdict(list)
    m_lCorrel50         = cl.defaultdict(list)
    m_lCorrel200        = cl.defaultdict(list)    
    
    def __init__(self, aTimeSeries, lSet):
        self.m_aTimeSeries  = aTimeSeries
        self.m_lSet         = lSet
        self.calcPairs()
        self.preprocess()
        
    def preprocess(self):
        print "Preprocess correlations"
        startTime = time.clock()
        self.m_lCorrel11.clear()        
        for t in xrange(len(self.m_aTimeSeries[self.m_aTimeSeries.keys()[0]])):
#             print ".",
            window = 11
            if t-window < 0:
                window = t
            for pair in self.m_lPairs:
                if window == 0:
                    self.m_lCorrel11[pair] += [0.0]
                else:
                    lFirst  = self.m_aTimeSeries[pair[0]][t-window:t]
                    lSecond = self.m_aTimeSeries[pair[1]][t-window:t]
                    correl  = self.getCorrel(lFirst, lSecond)
                    self.m_lCorrel11[pair] += [correl]
        print "doing..."
#         print "\n"
        self.m_lCorrel50.clear()        
        for t in xrange(len(self.m_aTimeSeries[self.m_aTimeSeries.keys()[0]])):
#             print ".",
            window = 50
            if t-window < 0:
                window = t
            for pair in self.m_lPairs:
                if window == 0:
                    self.m_lCorrel50[pair] += [0.0]
                else:
                    lFirst  = self.m_aTimeSeries[pair[0]][t-window:t]
                    lSecond = self.m_aTimeSeries[pair[1]][t-window:t]
                    correl  = self.getCorrel(lFirst, lSecond)
                    self.m_lCorrel50[pair] += [correl]
        print "doing..."
        print "\n"
        self.m_lCorrel200.clear()        
        for t in xrange(len(self.m_aTimeSeries[self.m_aTimeSeries.keys()[0]])):
#             print ".",
            window = 200
            if t-window < 0:
                window = t
            for pair in self.m_lPairs:
                if window == 0:
                    self.m_lCorrel200[pair] += [0.0]
                else:
                    lFirst  = self.m_aTimeSeries[pair[0]][t-window:t]
                    lSecond = self.m_aTimeSeries[pair[1]][t-window:t]
                    correl  = self.getCorrel(lFirst, lSecond)
                    self.m_lCorrel200[pair] += [correl]
        print "\n"
        print "\nPreprocessing finished (time: " + str(time.clock() - startTime) + ")"
    
    def calcPairs(self):
        lAllStocks  = self.m_aTimeSeries.keys()        
        self.m_lPairs = []
        for i in xrange(len(lAllStocks)):
            #break #uncomment to exclude correlations and speed up for debugging
            for j in xrange(i+1, len(lAllStocks)):
                    self.m_lPairs.append((lAllStocks[i],lAllStocks[j]))
        self.calcPairsRelated(self.m_lSet)       
    
    def calcPairsRelated(self, lSet):
        self.m_lSet = lSet
        self.m_lPairsRelated = []
        lMarket     = [item for item in self.m_aTimeSeries.keys() if item not in self.m_lSet]
        for x in lSet:
            #break #uncomment to exclude correlations and speed up for debugging
            for y in lMarket:
                if (x,y) in self.m_lPairs:
                    self.m_lPairsRelated.append((x,y))
                else:
                    self.m_lPairsRelated.append((y,x))
        # for x in lSet:
        #    for y in lSet:
        #        if not x == y:
        #            if (x,y) in self.m_lPairs:
        #                self.m_lPairsRelated.append((x,y))
        #            else:
        #                self.m_lPairsRelated.append((y,x))
    
    def changeSet(self, lSet):
        self.calcPairsRelated(lSet)
    
    def getMean(self, lX):
        mean = 0.0
        for x in lX:
            mean += x
        return mean / float(len(lX))
    
    def getVariance(self, lX):
        xmean = self.getMean(lX)
        xvar = 0.0
        for i in range(0, len(lX)):
            xvar += (lX[i]-xmean)*(lX[i]-xmean)
        xvar = xvar / float(len(lX))
        return xvar
        
    def getCorrel(self, lX, lY):
        if not len(lX) == len(lY):
            print "Error: correlation of two lists of different length: "+str(len(lX))+" <> "+str(len(lY))
        if len(lX) == 1:
            return 1.0
        xmean = self.getMean(lX)
        ymean = self.getMean(lY)
        xvar = 0.0
        yvar = 0.0
        cov = 0.0
        for i in range(0, len(lX)):
            cov  += (lX[i]-xmean)*(lY[i]-ymean)
            xvar += (lX[i]-xmean)*(lX[i]-xmean)
            yvar += (lY[i]-ymean)*(lY[i]-ymean)
        correl = cov / math.sqrt(xvar*yvar)
        return correl
    
    def getCorrelsForPairs(self, t, lPairs, window):
        aCorrels = None
        if window == 11:
            aCorrels = self.m_lCorrel11
        elif window == 50:
            aCorrels = self.m_lCorrel50
        elif window == 200:
            aCorrels = self.m_lCorrel200
        else:
            print "Invalid Correlation Window"
        aResult = []
        for pair in lPairs:
            aResult += [aCorrels[pair][t]]
        return aResult
    
    def getCorrels(self, t, window):
        return self.getCorrelsForPairs(t, self.m_lPairs, window)

    def getCorrelsRelated(self, t, window):
        return self.getCorrelsForPairs(t, self.m_lPairsRelated, window)
    
    def getWeightedCorrels(self, t, window):
        aCorrels = None
        if window == 11:
            aCorrels = self.m_lCorrel11
        elif window == 50:
            aCorrels = self.m_lCorrel50
        elif window == 200:
            aCorrels = self.m_lCorrel200
        else:
            print "Invalid Correlation Window"
        aResult = []
        for pair in self.m_lPairs:
            aResult += [aCorrels[pair][t]*self.m_aTimeSeries[pair[0]][t]]
            aResult += [aCorrels[pair][t]*self.m_aTimeSeries[pair[1]][t]]
        return aResult
    
    def getWeightedCorrelsRelated(self, t, window):
        aCorrels = None
        if window == 11:
            aCorrels = self.m_lCorrel11
        elif window == 50:
            aCorrels = self.m_lCorrel50
        elif window == 200:
            aCorrels = self.m_lCorrel200
        else:
            print "Invalid Correlation Window"
        aResult = []
        for pair in self.m_lPairsRelated:
            aResult += [aCorrels[pair][t]*self.m_aTimeSeries[pair[1]][t]]
        return aResult
          
if __name__ == "__main__":
#     aTargetStocks = ["ETR:ADS", "ETR:ALV", "ETR:BAS", "ETR:BEI", "ETR:BMW", "ETR:CBK", "ETR:DAI", "ETR:DB1", "ETR:DBK", "ETR:DPW", "ETR:DTE", "ETR:EOAN", "ETR:FME", "ETR:FRE", "ETR:HEI", "ETR:HEN3", "ETR:IFX", "ETR:LHA", "ETR:LIN", "ETR:MEO", "ETR:MRK", "ETR:MUV2", "ETR:RWE", "ETR:SAP", "ETR:SDF", "ETR:SIE", "ETR:TKA", "ETR:VOW3"]
#     aTargetStocks = ["ETR:ADS"] 
#     st = StockPriceFeatureStream(dataset = StockPriceFeatureStream.DAX30, targetStock=aTargetStocks, parameters="a1", nodes="4", label="annotation_DAX30_2007_2011_eps_short")
#     st = StockPrices(1)
    st  = DriftStockPrices(nodes = 5, driftProb = 0.001, numberOfStocks = 3, normalize=False)
    for idx in xrange(10):
        record, label = st.generate_example()
        print '******************'
        print "rec: " + str(record)
        print "reclen: " + str(len(record))
        print "label: " + str(label)
        
        
        
