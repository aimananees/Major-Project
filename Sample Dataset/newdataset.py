import pandas as pd
df = pd.read_csv('Indiabulls Integrated Services Limited.csv')

sma12 = df['Close'].rolling(13).mean()[12:]
sma26 = df['Close'].rolling(27).mean()[26:]

ema12_mul = 2/float(12+1)
ema26_mul = 2/float(26+1)

ema12_init = sma12[12] 
ema26_init = sma26[26] 

ema12 = pd.DataFrame(columns=['EMA12'])
ema12.loc[0] = ema12_init
k = 1
for x in df['Close'][13:]:
	ema12.loc[k]= (x - ema12.loc[k-1])*ema12_mul + ema12.loc[k-1]
	k+=1 

ema26 = pd.DataFrame(columns=['EMA26'])
ema26.loc[0] = ema26_init

k = 1
for x in df['Close'][27:]:
	ema26.loc[k]= (x - ema26.loc[k-1])*ema26_mul + ema26.loc[k-1]
	k+=1 


MACD = pd.DataFrame(columns=['MACD'])
for k in range(14,ema12.shape[0]):
	MACD.loc[k] = ema12.values[k] - ema26.values[k-14] 

signal = pd.DataFrame(columns=['Signal'])
signal.loc[0] = MACD[:9].mean().values[0]
signal_mul = 2/float(9+1)

k = 1
for x in MACD['MACD'][10:]:
	signal.loc[k]= (x - signal.loc[k-1])*signal_mul + signal.loc[k-1]
	k+=1

MACD_Hist = pd.DataFrame(columns=['MACD_Histogram'])
for k in range(9,MACD.shape[0]):
	MACD_Hist.loc[k] = MACD.values[k] - signal.values[k-9] 

#print MACD_Hist

typlist = ['High','Low','Close']
df['Typical'] = df[typlist].mean(axis=1)

sma20 = df['Typical'].rolling(21).mean()[20:]

mad = pd.DataFrame(columns=['Mean Average Deviation'])

for x in sma20.index:
	s = 0
	for i in range(x-20,x):
		s+=abs(sma20.values[x-20]-df['Typical'].values[i])
	mad.loc[x] = s/float(20)

k=0
cci = pd.DataFrame(columns=['CCI'])
for x in range(sma20.shape[0]):
	cci.loc[k] =  (df['Typical'].values[x] - sma20.values[x])/(0.015*mad.values[x])
	k+=1

#print cci	
so = pd.DataFrame(columns=['Stochastic Oscillator %k'])

for x in range(14,df.shape[0]):
	so.loc[x] = 100* (df['Close'].values[x] - min(df['Low'].values[x-14:x]))/(max(df['High'].values[x-14:x])-min(df['Low'].values[x-14:x]))

so_d = so['Stochastic Oscillator %k'].rolling(4).mean()[3:]
print so_d

#print df	
#print so
