

##################### Step 1 : Loading the train dataset and print the head. #####################


import pandas as pread
import numpy as nump
import matplotlib.pyplot as matPlot
import seaborn as sborn
trainFile = pread.read_json("train.json")
print(trainFile.head())



################### Printing number of instances and attributes #####################

print(trainFile.shape)

###################Metadata about train dataset###################
print(trainFile.describe())

################### Interest Level #####################

#################### Analysing the interest levels###################
interestLvl = ["low","medium","high"]
totalCount = trainFile.interest_level.value_counts().values
range1=[0.1,0,0]
effectColor = [  'lightblue','green','lightyellow']
patches, texts,autotexts= matPlot.pie(totalCount, labels=interestLvl,colors=effectColor,explode=range1,autopct="%1.1f%%",
                        startangle=45)
matPlot.title("Interest Level")
matPlot.show()

#Analysis from above: Low interest level is present as the majority. High interest is present in a very small population of dataset.


################ Price ##################

matPlot.figure(figsize=(9,9))
matPlot.scatter(range(trainFile.shape[0]),trainFile["price"].values,color='blue')
matPlot.title("Price Distribution")
matPlot.interactive(False)
matPlot.show()

####################From the graph, we can deduce that there are some outliers in Price.###################

####################Removing the outliers in price and replotting the graph.###################

greatest = nump.percentile(trainFile.price.values, 99)
trainFile['price'].ix[trainFile['price']>greatest] = greatest


matPlot.figure(figsize=(9,9))
matPlot.scatter(range(trainFile.shape[0]), trainFile["price"].values,color='blue')
matPlot.ylabel("Price")
matPlot.xlabel("Listings Count")
matPlot.title("Price Distribution");
matPlot.interactive(False)
matPlot.show()


####################Analysing Bathrooms###################


countBR = trainFile['bathrooms'].value_counts()

matPlot.figure(figsize=(6,6))
sborn.barplot(countBR.index, countBR.values, alpha=0.8, color="green")
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Bath Rooms', fontsize=10)
matPlot.show()

####################Analysis: So most of the listings have one bathroom.###################

####################Removing outliers in Bathrooms. Setting max as 3###################
trainFile['bathrooms'].ix[trainFile['bathrooms']>3] = 3

####################Analysing BedRooms###################
countBR = trainFile['bedrooms'].value_counts()

matPlot.figure(figsize=(6,6))
sborn.barplot(countBR.index, countBR.values, alpha=0.8, color="green")
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Bed Rooms', fontsize=10)
matPlot.show()

###################Analysis shows most of the listings have 1 Bed Room###################

###################Interest Level vs BathRooms###################

matPlot.figure(figsize=(9,9))
sborn.countplot(x='bathrooms', hue='interest_level', data=trainFile)
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Number of BathRooms', fontsize=10)
matPlot.show()

###################Analysis : Listings with one Bath Room has got very low interest.###################

####################Interest Level vs Bed Rooms###################

matPlot.figure(figsize=(9,9))
sborn.countplot(x='bedrooms', hue='interest_level', data=trainFile)
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Number of Bed Rooms', fontsize=10)
matPlot.show()

###################Analysis : Listings with one Bed Room has got very low interest.###################

###################Analysing the Latitude###################

lowerLmt = nump.percentile(trainFile.latitude.values, 1)
upperLmt = nump.percentile(trainFile.latitude.values, 99)
trainFile['latitude'].ix[trainFile['latitude']<lowerLmt] = lowerLmt
trainFile['latitude'].ix[trainFile['latitude']>lowerLmt] = lowerLmt

matPlot.figure(figsize=(8,8))
sborn.distplot(trainFile.latitude.values, bins=50, kde=False)
matPlot.xlabel('Plotting Latitude', fontsize=14)
matPlot.show()


####################Analysing Longitude###################



lowerLmt = nump.percentile(trainFile.latitude.values, 1)
upperLmt = nump.percentile(trainFile.latitude.values, 99)
trainFile['longitude'].ix[trainFile['longitude']<lowerLmt] = lowerLmt
trainFile['longitude'].ix[trainFile['longitude']>lowerLmt] = lowerLmt

matPlot.figure(figsize=(8,8))
sborn.distplot(trainFile.latitude.values, bins=50, kde=False)
matPlot.xlabel('Plotting Longitude', fontsize=14)
matPlot.show()

####################So we can deduce that longitude values are mostly in the range [-73.8 , -74.02].



####################Extracting Year,Month,Day from Created###################
trainFile["created"] = pread.to_datetime(trainFile["created"])
trainFile["createdDate"] = trainFile["created"].dt.date
trainFile["createdYear"]= trainFile["created"].dt.year
trainFile["createdMonth"]= trainFile["created"].dt.month
trainFile["createdDay"]= trainFile["created"].dt.day

####################Converting photos and features to numeric form###################
trainFile['photoCount'] = trainFile['photos'].apply(len)
trainFile['featureCount'] = trainFile['features'].apply(len)


###################Converting description to number of words###################
trainFile['numberOfWords'] = trainFile['description'].apply(lambda x: len(x.split(' ')))



trainFile = trainFile.drop(["created","description","display_address","features","photos","street_address"], axis=1)
trainFile.to_csv("resultstrain.csv")