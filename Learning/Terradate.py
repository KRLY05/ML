import teradata
import numpy as np
import pandas as pd
import time

#Time-cheking module
start_time = time.clock()
def runtime():
    time_spend=time.clock() - start_time
    print (time_spend, "seconds")
    
#Establishing connection to database
udaExec = teradata.UdaExec (appName="test", version="1.0",logConsole=False)
print ('udaExec - ok. spend ', end="")
runtime()

session = udaExec.connect(method="odbc", system="icwlive.uk.ba.com",
      username="gutabu02", password="Af7kw23e");
print ('session - ok. spend ', end="")
runtime()

#Running query
query="SELECT ROUTE_LVL1_DESC, ROUTE_LVL2_DESC,ROUTE_LVL3_DESC FROM LDB_SBOX_FINANCE03.V_TRAFFIC_STATS_KPIS sample 10"
result = pd.read_sql(query, session)
print ('results - ok. spend ', end="")
runtime()

#Saving data to file
result.to_csv('result.csv')
print ('data saved to result.csv spend ', end="")
runtime()

#Reading data from file
file_data=pd.read_csv('result.csv')
print(file_data)
print('done. spend ', end="")
runtime()
