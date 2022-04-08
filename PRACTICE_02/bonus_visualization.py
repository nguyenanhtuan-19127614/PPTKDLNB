import pandas as pd
import matplotlib.pyplot as plt
import seaborn  as sns
# read csv file and load to data frame
population = pd.read_csv('data/Countries Population from 1995 to 2020.csv')
population.info()

def top_30_bar_plot(year):
    # year = year you want to show
    # Get data top 30 population data of year
    current_population = population[population['Year'] == year][:30]
    # Fig size
    plt.rcParams['figure.figsize'] = (15, 7)
    ax = sns.barplot(x=current_population['Country'][:30], y=current_population['Population'][:30], palette='dark')
    #set x,y label
    ax.set_xlabel(xlabel='Countries', fontsize=10)
    ax.set_ylabel(ylabel='Population in Billion', fontsize=10)
    #set tittle
    title = 'Population of top 30 countries in ' + str(year)
    ax.set_title(label=title, fontsize=20)
    plt.xticks(rotation=90)
    plt.show()
def top_share_pie_plot(year,top):
    # year = year you want to show
    # top = top n country you want to show
    # get all country
    unique_countries = population['Country'].unique()
    #change plot type
    plt.style.use("seaborn-talk")
    # set year
    df_last_year = population[population['Year'] == year]
    series_last_year = df_last_year.groupby('Country')['Population'].sum().sort_values(ascending=False)
    labels = []
    values = []
    country_count = top # top country
    other_total = 0
    #count the top number, all countries below top 10 will add to the "other"
    for country in series_last_year.index:
        if country_count > 0:
            labels.append(country)
            values.append(series_last_year[country])
            country_count -= 1
        else:
            other_total += series_last_year[country]
    labels.append("Other")
    values.append(other_total)

    #Edit pie plot
    wedge_dict = {
        'edgecolor': 'black',
        'linewidth': 2
    }


    explode = (0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    plt.title(f"Total percent of World's Population \n the top 10 countries in {year}")
    plt.pie(values, labels=labels, explode=explode, autopct='%1.2f%%', wedgeprops=wedge_dict)
    plt.show()
top_share_pie_plot(2020,10)