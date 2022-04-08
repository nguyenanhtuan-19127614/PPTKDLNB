import pandas as pd
import matplotlib.pyplot as plt
PATH="C:/Users/Public/Desktop/"
# read csv file and load to data frame
df = pd.read_csv('data/covid-19-cases.csv')

# group by country column and sum over the different states/regions of each country
grouped = df.groupby('Country/Region')
#print(grouped)
df_countries = grouped.sum()

def make_column_plot(country):
    """make the bar plot of case numbers and change in numbers line plot."""

    # extract the series corresponding to the case numbers for country
    c_df = df_countries.loc[country, df_countries.columns[3:]]
    # convert index to a proper datetime object
    c_df.index = pd.to_datetime(c_df.index)
    # get the number of column
    n = len(c_df)
    # setting bar with corresponding values
    plt.bar(range(n), c_df.values)
    # setting x and y label
    plt.xlabel('Days')
    plt.ylabel('Confirmed cases, $N$')
    # add a title reporting the latest number of cases available
    title = '{}\n{} cases on {}'.format(country, c_df[-1],
                                        c_df.index[-1].strftime('%d %B %Y'))
    plt.suptitle(title)
    plot = plt.show()

def make_line_plot(country):
    #sort data from country
    c_df = df_countries.loc[country, df_countries.columns[3:]]
    # convert index to a proper datetime object
    c_df.index = pd.to_datetime(c_df.index)
    #make Line plot
    line_plt = c_df.plot(kind="line")
    #Set name for x and y label
    plt.xlabel('Days')
    plt.ylabel('Confirmed cases, $N$')
    # add a title reporting the latest number of cases available
    title = 'Line plot for {} Covid cases' \
            '\n{} cases on {} '.format(country, c_df[-1],
                                       c_df.index[-1].strftime('%d %B %Y'))
    plt.suptitle(title)
    plt.show()


def make_area_plot(country):
    # sort data from country
    c_df = df_countries.loc[country, df_countries.columns[3:]]
    # convert index to a proper datetime object
    c_df.index = pd.to_datetime(c_df.index)
    #make Area plot
    area_plt = c_df.plot(kind="area")
    # Set name for x and y label
    plt.xlabel('Days')
    plt.ylabel('Confirmed cases, $N$')
    # add a title reporting the latest number of cases available
    title = 'Area plot for {} Covid cases' \
            '\n{} cases on {} '.format(country, c_df[-1],
                                        c_df.index[-1].strftime('%d %B %Y'))
    plt.suptitle(title)
    plt.show()

def make_barh_plot(country):
    # sort data from country
    c_df = df_countries.loc[country, df_countries.columns[3:]]
    # convert index to a proper datetime object
    c_df.index = pd.to_datetime(c_df.index)
    # get the number of column
    n = len(c_df)
    # setting bar with corresponding values
    plt.barh(range(n), c_df.values)
    # setting x and y label
    plt.ylabel('Days')
    plt.xlabel('Confirmed cases, $N$')
    # add a title reporting the latest number of cases available
    title = 'Bar Horizontal plot for {} Covid cases' \
            '\n{} cases on {}'.format(country, c_df[-1],
                                        c_df.index[-1].strftime('%d %B %Y'))
    plt.suptitle(title)
    plt.show()

# make plot for a corresponding country

make_line_plot('Vietnam')
make_area_plot('Vietnam')
make_barh_plot('Vietnam')

