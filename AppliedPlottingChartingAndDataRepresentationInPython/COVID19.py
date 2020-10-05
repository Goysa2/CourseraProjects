import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


## Import COVID19 data from the new york times
## Data provided by the New York Times
## https://data.world/liz-friedman/us-covid-19-data-from-nytimes/workspace/file?filename=us-counties.csv
## https://data.world/liz-friedman/us-covid-19-data-from-nytimes/workspace/file?filename=us-states.csv
## https://data.world/liz-friedman/us-covid-19-data-from-nytimes/workspace/file?filename=us.csv
## https://github.com/nytimes/covid-19-data/blob/master/us-states.csv
## https://github.com/nytimes/covid-19-data/blob/master/us-counties.csv
## https://github.com/nytimes/covid-19-data/blob/master/us.csv
counties = pd.read_csv('us-counties.csv')
states = pd.read_csv('us-states.csv')
us = pd.read_csv('us.csv')

## Extract relevant data from orginal data frame for the county the state and the country
counties = counties.drop(columns='fips')
washtenaw = counties[counties['county'] == 'Washtenaw']
washtenaw['% death'] = (washtenaw['deaths']/washtenaw['cases']) * 100.0
washtenaw = washtenaw.drop(columns=['county', 'state'])
washtenaw['date'] = pd.to_datetime(washtenaw['date'])


michigan = states[states['state'] == 'Michigan']
michigan = michigan.drop(columns='fips')
michigan['% death'] = (michigan['deaths']/michigan['cases']) * 100.0
michigan['date'] = pd.to_datetime(michigan['date'])
michigan = michigan.drop(columns='state')

us['% death'] = (us['deaths']/us['cases']) * 100.0
us['date'] = pd.to_datetime(us['date'])
us = us[us['date'] > '2020-02-20']


## Plot the data
size_n = 5 # useful later

ax1 = washtenaw.plot('date', '% death', color = 'red', linewidth = size_n)#, xlabel = '', ylabel = 'Percentage of COVID19 patients who died')
ax1.xaxis.set_minor_locator(ticker.NullLocator())
ax1.set_title('Percentage of COVID19 cases resulting in death in Washtenaw county \n compared to Michigan and the rest of the USA', fontsize = 20)
ax1.set_xlabel('')
ax1.set_ylabel('Percentage of COVID19 cases resulting in death')
ax1.axvline('2020-03-10', color = 'gray')
ax1.text('2020-03-10', 8, '  First recorded case of COVID in Michigan \n March 10th, 2020')
ax2 = michigan.plot('date', '% death', color = 'black', ax = ax1, linewidth = size_n)
ax2.xaxis.set_minor_locator(ticker.NullLocator())
ax2.set_xlabel('')
ax3 = us.plot('date', '% death', color = 'gray', ax = ax1, linewidth = size_n)
ax3.xaxis.set_minor_locator(ticker.NullLocator())
ax3.set_xlabel('')
plt.legend(['% of death in Washtenaw county','% of death in Michigan', '% of death in the USA'],loc='lower right', frameon=False, fontsize= 25)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)


plt.show()
