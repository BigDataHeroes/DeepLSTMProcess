# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 21:04:37 2018

@author: David.Villaba
"""

from math import sqrt
from numpy import concatenate
#from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
 
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	agg = concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
TratarCentro=False
TratarArganzuela=False
TratarRetiro=False
TratarSalamanca=True

if (TratarCentro):
    dataset = read_csv('DatasetIndicadoresFinalCentro.csv', sep=';',index_col=0)
    dataset.index.name = 'anyo'
    vFirstIteration = True
    for vColumn in dataset.columns:
        if (vColumn == 'Precio'):
            continue
        
        if (vFirstIteration):
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
            vFirstIteration=False
        else:
            dataset = dataset[:-1]
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
            
        values = dataset.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(scaled, 1, 1)
        reframed.drop(reframed.columns[[150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297]], axis=1, inplace=True)
         
        values = reframed.values
        train = values[:7, :]
        test = values[7:, :]
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
         
        model = Sequential()
        model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
         
        vPercent=(inv_yhat[2]*100)/inv_y[2]
        vPercent = -100+vPercent
        vResultado ='PARA UN AUMENTO DEL 30% DE: '+ str(vColumn) + ', EL PRECIO DEL ALQUILER PREVISTO SERÁ: '+ str(inv_yhat[2]) + '(PRECIO ACTUAL: '+str(inv_y[2]) +'), AUMENTO: '+str(vPercent)+'%  \n'
        vResultadoCsv=str(vColumn) + ';'+ str(inv_yhat[2]) + ';'+str(inv_y[2]) +';'+str(vPercent)+'\n'
        with open("ResultadosCentro.txt", "a") as myfile:
            myfile.write(vResultado)
            
        with open("ResultadosCentro.csv", "a") as myfile:
            myfile.write(vResultadoCsv)
            
        print(str('GESTIONADO CENTRO: '+ str(vColumn)))
        
        
if (TratarArganzuela):
    dataset = read_csv('DatasetIndicadoresFinalArganzuela.csv', sep=';',index_col=0)
    dataset.index.name = 'anyo'
    vFirstIteration = True
    for vColumn in dataset.columns:
        if (vColumn == 'Precio'):
            continue
        
        if (vFirstIteration):
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
            vFirstIteration=False
        else:
            dataset = dataset[:-1]
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
        
        values = dataset.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(scaled, 1, 1)
        reframed.drop(reframed.columns[[150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298]], axis=1, inplace=True)
         
        values = reframed.values
        train = values[:7, :]
        test = values[7:, :]
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
         
        model = Sequential()
        model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
         
        vPercent=(inv_yhat[2]*100)/inv_y[2]
        vPercent = -100+vPercent
        vResultado ='PARA UN AUMENTO DEL 30% DE: '+ str(vColumn) + ', EL PRECIO DEL ALQUILER PREVISTO SERÁ: '+ str(inv_yhat[2]) + '(PRECIO ACTUAL: '+str(inv_y[2]) +'), AUMENTO: '+str(vPercent)+'%  \n'
        vResultadoCsv=str(vColumn) + ';'+ str(inv_yhat[2]) + ';'+str(inv_y[2]) +';'+str(vPercent)+'\n'
        with open("ResultadosArganzuela.txt", "a") as myfile:
            myfile.write(vResultado)
            
        with open("ResultadosArganzuela.csv", "a") as myfile:
            myfile.write(vResultadoCsv)
            
        print(str('GESTIONADO Arganzuela: '+ str(vColumn)))
        
if (TratarRetiro):
    dataset = read_csv('DatasetIndicadoresFinalRetiro.csv', sep=';',index_col=0)
    dataset.index.name = 'anyo'
    vFirstIteration = True
    for vColumn in dataset.columns:
        if (vColumn == 'Precio'):
            continue
        
        if (vFirstIteration):
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
            vFirstIteration=False
        else:
            dataset = dataset[:-1]
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
        
        values = dataset.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(scaled, 1, 1)
        reframed.drop(reframed.columns[[150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298]], axis=1, inplace=True)
         
        values = reframed.values
        train = values[:7, :]
        test = values[7:, :]
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
         
        model = Sequential()
        model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
         
        vPercent=(inv_yhat[2]*100)/inv_y[2]
        vPercent = -100+vPercent
        vResultado ='PARA UN AUMENTO DEL 30% DE: '+ str(vColumn) + ', EL PRECIO DEL ALQUILER PREVISTO SERÁ: '+ str(inv_yhat[2]) + '(PRECIO ACTUAL: '+str(inv_y[2]) +'), AUMENTO: '+str(vPercent)+'%  \n'
        vResultadoCsv=str(vColumn) + ';'+ str(inv_yhat[2]) + ';'+str(inv_y[2]) +';'+str(vPercent)+'\n'
        with open("ResultadosRetiro.txt", "a") as myfile:
            myfile.write(vResultado)
            
        with open("ResultadosRetiro.csv", "a") as myfile:
            myfile.write(vResultadoCsv)
            
        print(str('GESTIONADO Retiro: '+ str(vColumn)))
        
if (TratarSalamanca):
    dataset = read_csv('DatasetIndicadoresFinalSalamanca.csv', sep=';',index_col=0)
    dataset.index.name = 'anyo'
    vFirstIteration = True
    for vColumn in dataset.columns:
        if (vColumn == 'Precio'):
            continue
        
        if (vFirstIteration):
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
            vFirstIteration=False
        else:
            dataset = dataset[:-1]
            new_data = DataFrame(dataset[-1:].values, columns=dataset.columns)
            new_data.rename(index={0:11},inplace=True)
            new_data.at[11, vColumn] = new_data.at[11, vColumn] * 1.3
            dataset = dataset.append(new_data)
        
        values = dataset.values
        values = values.astype('float32')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        reframed = series_to_supervised(scaled, 1, 1)
        reframed.drop(reframed.columns[[150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298]], axis=1, inplace=True)
         
        values = reframed.values
        train = values[:7, :]
        test = values[7:, :]
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
         
        model = Sequential()
        model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.compile(loss='mae', optimizer='adam')
        history = model.fit(train_X, train_y, epochs=100, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        
        yhat = model.predict(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
         
        vPercent=(inv_yhat[2]*100)/inv_y[2]
        vPercent = -100+vPercent
        vResultado ='PARA UN AUMENTO DEL 30% DE: '+ str(vColumn) + ', EL PRECIO DEL ALQUILER PREVISTO SERÁ: '+ str(inv_yhat[2]) + '(PRECIO ACTUAL: '+str(inv_y[2]) +'), AUMENTO: '+str(vPercent)+'%  \n'
        vResultadoCsv=str(vColumn) + ';'+ str(inv_yhat[2]) + ';'+str(inv_y[2]) +';'+str(vPercent)+'\n'
        with open("ResultadosSalamanca.txt", "a") as myfile:
            myfile.write(vResultado)
            
        with open("ResultadosSalamanca.csv", "a") as myfile:
            myfile.write(vResultadoCsv)
            
        print(str('GESTIONADO Salamanca: '+ str(vColumn)))

    
print('FINALMENTE')
print('==========')
