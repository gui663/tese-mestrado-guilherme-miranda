import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#Grafico de barras da distribuição demografica
nf1 = (2,5)
wt = (6,4)

width = 0.15
fig = plt.figure()
plt.bar((0,0.35), wt, width, color="#c7c7c7")
plt.bar((0,0.35), nf1, width, bottom=wt, color="#12265f")

plt.xlabel('Sexo')
plt.ylabel('Quantidade')
plt.xticks((0,0.35), ('Macho', 'Fêmea'))
plt.yticks(np.arange(0, 12, 1))
plt.legend(labels=['WT', 'NF1'])
plt.show()

#Grafico da distribuição das epochs
df = pd.read_csv('./CSV/05092022_220542.csv')
names = df.loc[:, "Name"]
names_count = names.value_counts()
names_count.plot(kind='pie', colormap="GnBu", legend=False)
plt.show()
print(names_count)

categories = df.loc[:, "Label"]
categories_count = categories.value_counts()
print(categories_count)
#labels = ('2174.1', '2174.3', '2174.4', '2174.6', '2202.0', '2202.1', '2202.2', '2202.3', '1650.1', '1653.1', '2091.0', '2091.1', '2091.2', '2141.0'
#            ,'2141.2', '2141.3', '2142.4', '2174.1', '2174.3', '2174.4', '2174.6')