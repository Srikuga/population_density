# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 21:34:18 2021

@author: Kuka
"""

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np


# Part: Load and prepare the geopandas dataframes
#################################################

city_gdf= gpd.read_file('C:\\USD\\DM\\Programming_Project_2\\data\\citiesx010g.shp')

county_gdf= gpd.read_file('C:\\USD\\DM\\Programming_Project_2\\cb_2018_us_county_500k\\cb_2018_us_county_500k.shp')

# 19-IA
# 46-SD
# 17-IL
# 27-MN
# 18-IN

# get only the data for the states: IA, SD, IL, MN, and IN

county_gdf = county_gdf[county_gdf['STATEFP'].isin(['19','46','17','27','18'])]
city_gdf = city_gdf[city_gdf['STATE'].isin(['IA','SD','IL','MN','IN'])]


# Droping the COUNTY and COUNTYFIPS columns in the cities dataframe.
city_gdf=city_gdf.drop(['COUNTY', 'COUNTYFIPS'], axis=1)

# cities which are of type “Civil” as recorded in the “Feature” column.
city_gdf=city_gdf.loc[(city_gdf['FEATURE']=='Civil')]

# Part: compute the area of each county
########################################



#rename the column 'NAME'  as COUNTIES AND CITIES 
county_gdf=county_gdf.rename(columns={'NAME':'COUNTIES'})
city_gdf = city_gdf.rename(columns ={'NAME':'CITIES'})

county_projected_gdf = county_gdf.to_crs('epsg:4087')

# Filter the Iowa state records
###########################################
county_projected_Iowa_gdf =county_projected_gdf[county_projected_gdf['STATEFP']=='19']
county_Iowa_gdf =county_gdf[county_gdf['STATEFP']=='19']

# draw all 99 counties of Iowa for Projected and unprojected geometry
#####################################################################
fig, axs = plt.subplots(2)
fig.suptitle('projected and unprojected geometry-Iowa')
county_projected_Iowa_gdf.plot(ax=axs[0])
county_Iowa_gdf.plot(ax=axs[1])


cities_projeted_gdf = city_gdf.set_crs('epsg:4269')
counties_projeted_gdf = county_gdf.to_crs('epsg:4269')

# area of each county in square meters
#######################################
counties_projeted_gdf['AREA_COUNTY']=counties_projeted_gdf['geometry'].area

# Use a spatial join to determine which county each city belongs to.
###################################################################
city_belongs_county_gdf =gpd.tools.sjoin(cities_projeted_gdf, counties_projeted_gdf,how='left',op='within')

# city_belongs_county_gdf.to_file('city_belongs_county.shp',driver='ESRI Shapefile')

# print to the screen: a list top five largest cities
#####################################################
largest_cities =cities_projeted_gdf.sort_values(by =['POP_2010'],ascending=False)
print('Top five largest cities :')
print(largest_cities['CITIES'].head(5))


# The total population of each county
########################################
total_population_gdf   = city_belongs_county_gdf.groupby(['COUNTIES'])['POP_2010'].sum()
print(total_population_gdf)

area_gdf=city_belongs_county_gdf.groupby('COUNTIES')['AREA_COUNTY'].mean()
# People per sq meter in each county
#####################################
print('People per sq meter in each county')
population_density = total_population_gdf/area_gdf
print(population_density)


# A count of the number of cities/towns within each county
##########################################################
gdf_city_count = city_belongs_county_gdf[['CITIES','COUNTIES']]
gdf_city_count = gdf_city_count.groupby('COUNTIES').count().rename(columns={'CITIES':'NUMBER_CITIES'})
print(gdf_city_count)

# print to the screen: a list top five most populous counties in each state
#############################################################################
gdf_counties_States = city_belongs_county_gdf[['STATE','POP_2010','COUNTIES']]
gdf_counties_States =gdf_counties_States.groupby(['STATE','COUNTIES']).sum()
sorted_gdf =gdf_counties_States.sort_values(by =['POP_2010'],ascending =False)
print(sorted_gdf.groupby('STATE').head(5))

# print to the screen: a list top five most populous counties across all five states
####################################################################################
print(sorted_gdf.head(5))

# print to the screen: a list the top five counties in terms of population density
#################################################################################
print('Population Density : ')
print(population_density.sort_values(ascending =False).head(5))


# Visualize the population of each county in IA
##############################################################


Pop_County_gdf = pd.DataFrame(city_belongs_county_gdf.groupby(['COUNTIES'])['POP_2010'].sum().reset_index())

counties = gpd.GeoDataFrame(Pop_County_gdf.merge(counties_projeted_gdf, how='left'))

counties = counties.loc[(counties['STATEFP']=='19')]

counties['log_POP_2010'] = np.log10(counties['POP_2010'])
counties['Per_POP_2010']=counties['POP_2010']/counties['POP_2010'].sum()

fig,ax = plt.subplots(2)
counties.plot(ax=ax[0],column='POP_2010', edgecolor="black",legend=True,legend_kwds={'label': "Population by Country"})
counties.plot(ax=ax[1],column='log_POP_2010', edgecolor="black",legend=True,legend_kwds={'label': "log value of Population by Country"})

fig.show()

patches = []

fig = plt.figure(figsize=(12,9))

ax1=fig.add_subplot(1,1,1)

minv, maxv = counties['POP_2010'].min(), counties['POP_2010'].max()
cmap = plt.cm.get_cmap('Reds')
norm = plt.Normalize(minv, maxv)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax1, label='Population')

# x=0
for i,v in counties['geometry'].iteritems():
    # print(i)
    poly=counties['geometry'][i]
    x_array, y_array = poly.exterior.coords.xy
    x_list = np.dstack(x_array).tolist()[0]
    y_list = np.dstack(y_array).tolist()[0]
    ax1.plot(x_list, y_list, color='black', linewidth=0.1, zorder=1)

    
    patches.append( Polygon(np.vstack((x_list, y_list)).T, True, color= cmap(norm(counties['POP_2010'][i])), linewidth=0.1))
    
pc = PatchCollection(patches,match_original=True,edgecolor='black',linewidths=0.1,zorder=2)
im = ax1.add_collection(pc)


