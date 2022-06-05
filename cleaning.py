# Shamelessly yeeted from here https://colab.research.google.com/drive/1TzC_cQIIhplKAd1ceHxABS9FFtFaoyMt 
# And modified to fit my needs
import numpy as np
import pandas as pd
import math
from datetime import datetime
import time
import IPython
import os
import threading
import csv
import warnings
warnings.filterwarnings('error', category=UserWarning)

##############################################################################
#GLOBAL VARIABLES
##############################################################################

ORDERDATES = {
    "Q1": [1645812000, 1646676000, 1647280800, 1647885600, 1648486800],
    "Q2": [1649091600, 1649350800, 1649696400, 1649955600, 1650301200, 1650560400, 1650906000, 1651165200, 1651510800, 1651770000, 1652115600, 1652374800, 1652720400, 1652979600, 1653325200, 1653584400, 1653930000, 1654189200, 1654534800, 1654794000, 1655139600, 1655398800, 1655744400, 1656003600, 1656349200, 1656608400],
    "Q3": [1656954000, 1657213200, 1657558800, 1657818000, 1658163600, 1658422800, 1658768400, 1659027600, 1659373200, 1659632400, 1659978000, 1660237200, 1660582800, 1660842000, 1661187600, 1661446800, 1661792400, 1662051600, 1662397200, 1662656400, 1663002000, 1663261200, 1663606800, 1663866000, 1664211600, 1664470800]
    }

ORDERDATES_MONDAYS = {
    "Q1": [1645812000, 1646676000, 1647280800, 1647885600, 1648486800],
    "Q2": [1649091600, 1649696400, 1650301200, 1650906000, 1651510800, 1652115600, 1652720400, 1653325200, 1653930000, 1654534800, 1655139600, 1655744400, 1656349200],
    "Q3": [1656954000, 1657558800, 1658163600, 1658768400, 1659373200, 1659978000, 1660582800, 1661187600, 1661792400, 1662397200, 1663002000, 1663606800, 1664211600]
    }
  
ALLDATES = []
for quarter, quarterDates in ORDERDATES.items():
  for d in quarterDates:
    ALLDATES.append(d)

ALLDATES = np.array(ALLDATES)
ALLDATES_ROUNDED = np.round(ALLDATES, -5)
#[1649091600 + (i % 2) * 604800 * i + (1 - (i % 2)) * 259200 * i for i in range(13)]
#[1656954000 + 604800*i for i in range(13)]

REGIONLIST = ["US", "EU", "UK", "ALL"]
MODELLIST = ["64", "256", "512"]
ESTIMATELIST = ["Q1", "Q2", "Q3"]

def AllPossibleQueues_Func():
  result = []
  for r in REGIONLIST:
    if (r == "ALL"):
      continue
    for m in MODELLIST:
      for e in ESTIMATELIST:
        result.append((r,m,e))
    
  return result

ALLPOSSIBLEQUEUES = AllPossibleQueues_Func()

def QueueIndex(df):
  return [ALLPOSSIBLEQUEUES.index((entry["Region"], entry["Model"], entry["ValveEstimate"])) for index, entry in df.iterrows()]

def BatchTimestamp(df):
  return [BatchTimestamp_OneEntry(entry)  for index, entry in df.iterrows()]

def BatchTimestamp_OneEntry(entry):
  if (np.isnan(entry["TrueOrderTimestamp"])):
    return np.nan
  
  for d in ALLDATES:
    if (np.round(entry["TrueOrderTimestamp"], -5) == np.round(d, -5)):
      return d

  return np.nan

##############################################################################
#IMPORTING DATA (DEFINIING THE FUNCTION)
##############################################################################

def ImportData():
  #take CSV from straight from moo's survey
  os.system("wget https://docs.google.com/spreadsheets/d/1QqlSUpqhyBCBYeu_gW4w5vIxfcd7qablSviALDFJ0Dg/gviz/tq?tqx=out:csv")
  

  global data

  #import csv
  data = pd.read_csv("tq?tqx=out:csv")

  ##############################################################################
  #CLEANING DATA
  ##############################################################################

  print(f"Cleaning data, starting from {len(data)} entries...\n")

  #cast Model column to string
  data = data.astype({"Model": str})

  #use Google Form's timestamp column to manually remove entries
  entries_to_smite = ['3/14/2022 15:04:24', '3/22/2022 10:06:34', '3/22/2022 15:25:45''4/11/2022 18:24:15',
                      '4/7/2022 15:34:19', '4/12/2022 7:43:29', '4/11/2022 22:32:08', '4/11/2022 22:28:21',
                      '4/7/2022 15:19:01', '3/7/2022 7:08:58', '4/12/2022 13:44:37', '4/13/2022 8:58:53',
                      '3/21/2022 19:36:56', '3/7/2022 15:39:38', '5/5/2022 18:43:15', '5/12/2022 12:27:49',
                      '5/3/2022 8:31:01', '4/19/2022 5:35:52']
  """
  
  """
  @np.vectorize
  def toSmite(googleTimestamp):
    return googleTimestamp in entries_to_smite
  
  data.rename(columns = {"Timestamp": "FormTimestamp"}, inplace=True)
  smittenMask = toSmite(data["FormTimestamp"])
  print(f"{np.sum(smittenMask)} entries smitten by jimmosio's hammer\n")
  data.drop(data[smittenMask].index, inplace=True)

  #remove unnecessary variables, rename useful ones
  data.drop(columns = ["Initial Valve Estimate", "Unnamed: 7"], inplace=True)
  data.rename(columns = {"rtReserveTime or preorder-email time ": "Timestamp", "Your most recent pre-order estimated time":"ValveEstimate", "When did you receive your ready to order email?":"TrueOrderTimestamp"}, inplace=True)

  #remove rows with missing values on vital fields
  namask = data["Region"].isna() | data["Model"].isna() | data["ValveEstimate"].isna() | data["Timestamp"].isna() | data["ValveEstimate"].isna()
  print(f"{np.sum(namask)} entries removed for having NA on vital fields\n")
  data.drop(data[namask].index,inplace=True)

  #not valid for after Q2
  #data = data[data["ValveEstimate"] != "After Q2"]
  notEstimable = np.vectorize(lambda x: x not in ESTIMATELIST)
  notEstimableMask = notEstimable(data["ValveEstimate"])
  print(f"{np.sum(notEstimableMask)} entries removed for being \"After Q2\", \"After Q3\", or similar\n")
  data.drop(data[notEstimableMask].index, inplace=True)
  
  #some timestamps may be in milliseconds, or nanoseconds
  #note: timestamps will realistically have 10 digits, everything else is either in ms, ns or wrong input
  @np.vectorize
  def digits(x):
    if (np.isnan(x)):
      return -1
    
    return int(np.log10(x))+1
  
  @np.vectorize
  def convertTimestampToSeconds(x):
    if (np.isnan(x)):
      return np.nan
    
    l = digits(x)
    if (l > 10 and (l - 10) % 3 == 0):
      m = (l - 10) // 3
      for _ in range(m):
        x = x // 1000
    elif (l != 10):
      x = np.nan
    
    return x

  data["Timestamp"] = convertTimestampToSeconds(data["Timestamp"])
  data["TrueOrderTimestamp"] = convertTimestampToSeconds(data["TrueOrderTimestamp"])

  #remove invalid entries

  #entries with a blatantly wrong "TrueOrderTimestamp"
  wrongTrueOrderMask = data["TrueOrderTimestamp"].notna() & ((np.round(data["Timestamp"], -7) >= np.round(data["TrueOrderTimestamp"], -7)) | (np.round(data["Timestamp"], -5) == np.round(data["TrueOrderTimestamp"], -5)))
  print(f"{np.sum(wrongTrueOrderMask)} entries removed for having the order email timestamp the same as the reserve time\n")
  data.drop(data[wrongTrueOrderMask].index, inplace=True)

  @np.vectorize
  def earliestDate(valveEstimate):
    return ORDERDATES[valveEstimate][0]
  
  @np.vectorize
  def latestDate(valveEstimate):
    return ORDERDATES[valveEstimate][-1]
  
  superEarlyOrderMask = data["TrueOrderTimestamp"].notna() & ((earliestDate(data["ValveEstimate"]) > data["TrueOrderTimestamp"]))
  print(f"{np.sum(superEarlyOrderMask)} entries removed for having an order email timestamp which is before the beginning of their quarter\n")
  data.drop(data[superEarlyOrderMask].index, inplace=True)

  superLateOrderMask = data["TrueOrderTimestamp"].notna() & (np.round(latestDate(data["ValveEstimate"]), -5) < np.round(data["TrueOrderTimestamp"], -5))
  print(f"{np.sum(superLateOrderMask)} entries removed for having an order email timestamp which is after the end of their quarter\n")
  data.drop(data[superLateOrderMask].index, inplace=True)

  #entries supposedly happening before july 16th 10am PDT
  cretinMask = data["Timestamp"] < 1626454800
  print(f"{np.sum(cretinMask)} entries removed for being before July 16th 10am PDT\n")
  data.drop(data[cretinMask].index, inplace=True)

  #attempt salvaging entries like "july 16 2022" by subtracting one year
  #yes, Q1 Q2 and Q3 were put there explicitely
  bruhMomentMask = ((data["ValveEstimate"] == "Q1") | (data["ValveEstimate"] == "Q2") | (data["ValveEstimate"] == "Q3")) & (data["Timestamp"] >= 1657990800)
  data.loc[bruhMomentMask, "Timestamp"] -= 31536000
  print(f"({np.sum(bruhMomentMask)} entries' reserve times are probably off by one year, they were shifted back)\n")

  #entries supposedly happening after feb 25th 10am PDT
  wrongUpdateMask = ((data["ValveEstimate"] == "Q1") | (data["ValveEstimate"] == "Q2") | (data["ValveEstimate"] == "Q3")) & (data["Timestamp"] >= 1631000000)
  print(f"{np.sum(wrongUpdateMask)} entries removed for being Q1, Q2 or Q3 and after ~September\n")
  data.drop(data[wrongUpdateMask].index, inplace=True)
  
  #drop everything whose timestamp doesn't have 10 digits
  wrongDigits = np.vectorize(lambda x: digits(x) != 10)
  wrongDigitsTimestampMask = wrongDigits(data["Timestamp"])
  wrongDigitsOrderTimestampMask = wrongDigits(data["TrueOrderTimestamp"]) & data["TrueOrderTimestamp"].notna()
  wrongDigitsMask = wrongDigitsTimestampMask | wrongDigitsOrderTimestampMask
  print(f"{np.sum(wrongDigitsMask)} entries removed for having timestamps with a strange number of digits\n")
  data.drop(data[wrongDigitsMask].index, inplace=True)

  #batch timestamp
  data["TrueOrderTimestamp"] = BatchTimestamp(data)

  #clip middle 95%, queue by queue
  M = 0
  clip_th = 0.025
  data["QueueIndex"] = QueueIndex(data)

  for i in range(len(ALLPOSSIBLEQUEUES)):
    queueIndexMask = (data["QueueIndex"] == i)
    if (np.sum(queueIndexMask) < 20):
      continue
    
    d = data[queueIndexMask]
    q_min = np.quantile(d["Timestamp"], clip_th)
    q_max = np.quantile(d["Timestamp"], 1 - clip_th)
    extremeValueMask = (data["Timestamp"] < q_min) | (data["Timestamp"] > q_max)
    m = queueIndexMask & extremeValueMask
    M += np.sum(m)
    #data.drop(data[m].index, inplace=True)
    data.loc[m, "Timestamp"] = np.clip(data.loc[m, "Timestamp"], q_min, q_max)

  if (M > 0):
    print(f"({M} entries clipped: timestamps outside the middle {(1 - 2*clip_th) * 100:.0f}% of observations, for each queue, were clipped to the boundary)\n")
  
  clip_th = 0.1
  N = 0
  for i in range(len(ALLPOSSIBLEQUEUES)):
    queueIndexMask = (data["QueueIndex"] == i)
    for d in np.round(ALLDATES, -5):
      mask = queueIndexMask & data["TrueOrderTimestamp"].notna() & (np.round(data["TrueOrderTimestamp"], -5) == d)
      
      if (np.sum(mask) < 20):
        continue
      
      d = data[mask]
      q_min = np.quantile(d["Timestamp"], clip_th)
      q_max = np.quantile(d["Timestamp"], 1 - clip_th)
      extremeValueMask = (data["Timestamp"] < q_min) | (data["Timestamp"] > q_max)
      n = mask & extremeValueMask
      N += np.sum(n)
      #data.drop(data[m].index, inplace=True)
      data.loc[n, "Timestamp"] = np.clip(data.loc[n, "Timestamp"], q_min, q_max)

  if (N > 0):
    print(f"({N} entries clipped: entries with order timestamps with reservation timestamps outside the middle {(1 - 2*clip_th) * 100:.0f}% of observations, for each queue and batch, had their reservation timestamp clipped to the boundary)\n")

  #something is wrong with this one, and i quite don't know what
  #possibly: massive skill issue on the side of respondents, or systemic overlap caused by a third factor
  """
  #HAVE THIS CLEANING STEP *AFTER* EVERY OTHER ONE (cause i use SubData())
  #remove illogical entries i.e. Q1 order that was apparently done on september 2021
  #entries of a given quarter must be all earlier than the earliest order of the next quarter
  S = 0
  for r in REGIONLIST:
    if (r == "ALL"):
      continue
    
    for m in MODELLIST:
      earliestFromLaterQuarter = -1
      
      for i in range(len(ESTIMATELIST)):
        e = ESTIMATELIST[-i-1]

        if (i > 0):
          mask = (data["Region"] == r) & (data["Model"] == m) & (data["ValveEstimate"] == e) & (data["Timestamp"] >= earliestFromLaterQuarter)
          s = np.sum(mask)
          if (s > 0):
            S += s
            print(f"{s} entries removed from {r} {m}GB {e} for having reserve times later than the earliest one next quarter\n")

          data.drop(data[mask].index, inplace=True)
          
        sb = SubData(r,m,e)
        earliestFromLaterQuarter = np.amin(sb["Timestamp"]);
        print(f"{r} {m}GB {e}: {earliestFromLaterQuarter}")
      
  print(f"{S} entries removed for having reserve times later than the earliest one next quarter\n")
  """
  print(f"DONE, dataset at {len(data)} entries")
  
  data.to_csv(f"filteredstuff.csv", index=False)

  print(f"Wrote cleaned data to filteredstuff.csv")


ImportData()