import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Hypothesis: University towns have their mean housing prices less effected by recessions. Run a t-test to compare
# the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the
# recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)

states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National',
          'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana',
          'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho',
          'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan',
          'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico',
          'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa',
          'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana',
          'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California',
          'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island',
          'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia',
          'ND': 'North Dakota', 'VA': 'Virginia'}


def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ],
    columns=["State", "RegionName"]  )

    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    file1 = open('university_towns.txt', 'r')
    lines = file1.readlines()
    lst = []
    state = ''
    region = ''
    for line in lines:
        if '[ed' in line:
            state = line.split('[ed')[0]
        else:
            if '(' in line:
                region0 = line.split('(')[0]
                region = region0.strip()
            elif '\n' in line:
                region = line.rstrip()
            lst.append([state, region])

    uni_town = pd.DataFrame(lst)
    uni_town.columns = ['State', 'RegionName']
    return uni_town

# ut = get_list_of_university_towns()

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a
    string value in a format such as 2005q3'''
    df = pd.read_excel('gdplev.xls', header = 5).reset_index().drop([0, 1])
    columns_to_keep = ['Unnamed: 4', 'GDP in billions of chained 2009 dollars.1']
    df = df[columns_to_keep]
    df = df.rename(columns = {'Unnamed: 4' : 'YearQuarter', 'GDP in billions of chained 2009 dollars.1' : 'GDP in billions of chained 2009 dollars'})
    df = df.reset_index()
    df = df[['YearQuarter', 'GDP in billions of chained 2009 dollars']]
    df = df[211 : -1].reset_index()
    df = df[['YearQuarter', 'GDP in billions of chained 2009 dollars']]
    nb_of_qt = len(df.index)
    recess_start = None
    for i in range(2,nb_of_qt):
        if df.loc[i]['GDP in billions of chained 2009 dollars'] < df.loc[i-1]['GDP in billions of chained 2009 dollars']:
            if df.loc[i-1]['GDP in billions of chained 2009 dollars'] < df.loc[i-2]['GDP in billions of chained 2009 dollars']:
                recess_start = df.loc[i-3]['YearQuarter']
    return recess_start

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a
    string value in a format such as 2005q3'''
    df = pd.read_excel('gdplev.xls', header = 5).reset_index().drop([0, 1])
    columns_to_keep = ['Unnamed: 4', 'GDP in billions of chained 2009 dollars.1']
    df = df[columns_to_keep]
    df = df.rename(columns = {'Unnamed: 4' : 'YearQuarter', 'GDP in billions of chained 2009 dollars.1' : 'GDP'})
    df = df.reset_index()
    df = df[['YearQuarter', 'GDP']]
    df = df[211 : -1].reset_index()
    df = df[['YearQuarter', 'GDP']]
    nb_of_qt = len(df.index)
    recess_start = None
    recess_end = None
    for i in range(2,nb_of_qt):
        if df.loc[i]['GDP'] < df.loc[i-1]['GDP']:
            if df.loc[i-1]['GDP'] < df.loc[i-2]['GDP']:
                recess_start = df.loc[i-2]['YearQuarter']
        if (recess_start is not None) & (recess_end is None):
            if df.loc[i]['GDP'] < df.loc[i+1]['GDP']:
                if df.loc[i+1]['GDP'] < df.loc[i+2]['GDP']:
                    recess_end = df.loc[i+2]['YearQuarter']
    return recess_end

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a
    string value in a format such as 2005q3'''
    df = pd.read_excel('gdplev.xls', header = 5).reset_index().drop([0, 1])
    columns_to_keep = ['Unnamed: 4', 'GDP in billions of chained 2009 dollars.1']
    df = df[columns_to_keep]
    df = df.rename(columns = {'Unnamed: 4' : 'YearQuarter', 'GDP in billions of chained 2009 dollars.1' : 'GDP'})
    df = df.reset_index()
    df = df[['YearQuarter', 'GDP']]
    df = df[211 : -1].reset_index()
    df = df[['YearQuarter', 'GDP']]
    nb_of_qt = len(df.index)
    recess_start = None
    recess_end = None
    df = df.set_index(['YearQuarter'])
    recess_start = get_recession_start()
    recess_end = get_recession_end()
    return pd.Series.idxmin(df[recess_start:recess_end]['GDP'])


def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].

    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.

    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    df = df.drop(['Metro', 'RegionID', 'CountyName', 'SizeRank'], axis=1)
    states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National',
              'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana',
              'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho',
              'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan',
              'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi',
              'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota',
              'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut',
              'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York',
              'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado',
              'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota',
              'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia',
              'ND': 'North Dakota', 'VA': 'Virginia'}
    df = df.replace({'State': states})
    df = df.set_index(['State', 'RegionName'])
    df.columns = pd.to_datetime(df.columns).to_period('M')
    col_to_keep = []
    for col in df.columns:
        if (type(col) == pd.Period) & (col.year >= 2000):
            col_to_keep.append(col)
    df = df[col_to_keep]
    df = df.resample('q', axis=1).mean()
    df = df.rename(columns=lambda x: str(x).lower())
    return df


def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values,
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence.

    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    uni_town = get_list_of_university_towns()
    homes = convert_housing_data_to_quarters()
    recess_start = get_recession_start()
    recess_end = get_recession_end()
    recess_bottom = get_recession_bottom()
    homes = homes.loc[:, recess_start:recess_bottom]
    homes['New Data'] = homes[recess_start] / homes[recess_bottom]
    housing_data_of_uni_town = pd.merge(homes, uni_town, how='inner', left_on=None, right_on=['State', 'RegionName'],
                                        left_index=True, right_index=False, indicator=True)
    housing_data_of_uni_town = housing_data_of_uni_town.drop(['_merge'], axis=1)
    housing_data_of_uni_town = housing_data_of_uni_town.set_index(['State', 'RegionName'])

    housing_data_of_non_uni_town = homes.drop(housing_data_of_uni_town.index, axis=0)
    stats = ttest_ind(housing_data_of_non_uni_town['New Data'], housing_data_of_uni_town['New Data'], nan_policy='omit')

    different = None
    better = None

    if stats.pvalue < 0.01:
        different = True
    else:
        different = False

    uni_town_ratio_mean = housing_data_of_uni_town['New Data'].mean()
    non_uni_town_ratio_mean = housing_data_of_non_uni_town['New Data'].mean()
    if uni_town_ratio_mean < non_uni_town_ratio_mean:
        better = 'university town'
    else:
        better = 'non-university town'
    return (different, stats.pvalue, better)


print(run_ttest())