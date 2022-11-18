#fazer código para dado um evento, devolver a janela de 200 ms que se segue
import pandas as pd
import preProcess as pp 

def get_window(event_timestamp, signal, chanel, last_i=0):
    #retorna a janela de 200ms após um determinado evento
    timestamp = signal.iloc[last_i:,0]
    timestamp = timestamp.values.tolist()
    offset = 200*10**3
    baseline_offset = 100*10*3

    offset_index = int(offset/100)
    baseline_offset_index = int(baseline_offset/100)
  
    for i in timestamp:
        
        if(i == event_timestamp or i == event_timestamp+50):
            index = timestamp.index(i)
            #timestamps = signal.iloc[index:index+offset_index, 0]
            try:
                if (index+offset_index<=len(signal)):
                    data = signal.iloc[index:index+offset_index, chanel]
                    #data = signal.iloc[index-baseline_offset_index:index+offset_index, chanel]
                    window = list(data)
                    return window, index
            except:
                
                return 0, last_i
            #window = pd.DataFrame(list(zip(timestamps, data)), columns=['TimeStamp [µs]', 'E4 (ID=3) [pV]'])
            

def get_epochs(signal, timestamps, chanel):

    data = []
    last_i = 0
    print(signal)
    signal.iloc[:,chanel] = pp.filters(signal.iloc[:,chanel])
    print(signal)
    for i in range(len(timestamps)):
        timestamp = timestamps.iloc[i, 0]
        window, last_i = get_window(timestamp, signal, chanel, last_i)
        #print(i)
        if window!=0:
            #print(i)
            data.append(window)
    if data:
        epochs = pd.DataFrame(data)   
        return epochs
    else: 
        return None


def load_epochs(path):

    epochs = pd.read_csv(path)

    return epochs

def average_epochs(epochs):

    num_values= len(epochs.columns)
    num_epochs = len(epochs)

    average_signal = []

    for i in range(num_values):
        sum = 0
        for k in range(num_epochs):
            sum += epochs.iloc[k, i]
        
        average = sum/num_epochs
        average_signal.append(average)
        #print(average_signal[i])


    return average_signal

