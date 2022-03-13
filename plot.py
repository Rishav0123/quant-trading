import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

df = pd.read_csv("pred_test.csv")
#print(df['Unnamed: 0'])
for cols in df.columns:
    print(cols)
#ax = plt.gca()
df['index'] = range(1, len(df) + 1)
#plot individual lines
print(df)
# Apply the default theme
sns.set_theme()



# Create a visualization
sns.lineplot('index', 'value', hue='variable',
             data=pd.melt(df, 'index'))
matplotlib.pyplot.show()
#plt.xticks(rotation = 25)


#display plot
#plt.show()
#df.plot(kind='line',x='Unnamed: 0',y='test', color='red', ax=ax)
