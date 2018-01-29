

##################### Step 1 : Loading the test dataset and print the head. #####################


import pandas as pread
import numpy as nump
import matplotlib.pyplot as matPlot
import seaborn as sborn
testFile = pread.read_json("test.json")
print(testFile.head())



################### Printing number of instances and attributes #####################

print(testFile.shape)

###################Metadata about test dataset###################
print(testFile.describe())

################### Interest Level #####################

#################### Analysing the interest levels###################
interestLvl = ["low","medium","high"]
totalCount = testFile.interest_level.value_counts().values
range1=[0.1,0,0]
effectColor = [  'lightblue','green','lightyellow']
patches, texts,autotexts= matPlot.pie(totalCount, labels=interestLvl,colors=effectColor,explode=range1,autopct="%1.1f%%",
                        startangle=45)
matPlot.title("Interest Level")
matPlot.show()

#Analysis from above: Low interest level is present as the majority. High interest is present in a very small population of dataset.


################ Price ##################

matPlot.figure(figsize=(9,9))
matPlot.scatter(range(testFile.shape[0]),testFile["price"].values,color='blue')
matPlot.title("Price Distribution")
matPlot.interactive(False)
matPlot.show()

####################From the graph, we can deduce that there are some outliers in Price.###################

####################Removing the outliers in price and replotting the graph.###################

greatest = nump.percentile(testFile.price.values, 99)
testFile['price'].ix[testFile['price']>greatest] = greatest


matPlot.figure(figsize=(9,9))
matPlot.scatter(range(testFile.shape[0]), testFile["price"].values,color='blue')
matPlot.ylabel("Price")
matPlot.xlabel("Listings Count")
matPlot.title("Price Distribution");
matPlot.interactive(False)
matPlot.show()



####################Analysing Bathrooms###################


countBR = testFile['bathrooms'].value_counts()

matPlot.figure(figsize=(6,6))
sborn.barplot(countBR.index, countBR.values, alpha=0.8, color="green")
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Bath Rooms', fontsize=10)
matPlot.show()

####################Analysis: So most of the listings have one bathroom.###################

####################Removing outliers in Bathrooms. Setting max as 3###################
testFile['bathrooms'].ix[testFile['bathrooms']>3] = 3

####################Analysing BedRooms###################
countBR = testFile['bedrooms'].value_counts()

matPlot.figure(figsize=(6,6))
sborn.barplot(countBR.index, countBR.values, alpha=0.8, color="green")
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Bed Rooms', fontsize=10)
matPlot.show()

###################Analysis shows most of the listings have 1 Bed Room###################

###################Interest Level vs BathRooms###################

matPlot.figure(figsize=(9,9))
sborn.countplot(x='bathrooms', hue='interest_level', data=testFile)
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Number of BathRooms', fontsize=10)
matPlot.show()

###################Analysis : Listings with one Bath Room has got very low interest.###################

####################Interest Level vs Bed Rooms###################

matPlot.figure(figsize=(9,9))
sborn.countplot(x='bedrooms', hue='interest_level', data=testFile)
matPlot.ylabel('Number of Instances', fontsize=10)
matPlot.xlabel('Number of Bed Rooms', fontsize=10)
matPlot.show()

###################Analysis : Listings with one Bed Room has got very low interest.###################

###################Analysing the Latitude###################

lowerLmt = nump.percentile(testFile.latitude.values, 1)
upperLmt = nump.percentile(testFile.latitude.values, 99)
testFile['latitude'].ix[testFile['latitude']<lowerLmt] = lowerLmt
testFile['latitude'].ix[testFile['latitude']>lowerLmt] = lowerLmt

matPlot.figure(figsize=(8,8))
sborn.distplot(testFile.latitude.values, bins=50, kde=False)
matPlot.xlabel('Plotting Latitude', fontsize=14)
matPlot.show()


####################Analysing Longitude###################



lowerLmt = nump.percentile(testFile.latitude.values, 1)
upperLmt = nump.percentile(testFile.latitude.values, 99)
testFile['longitude'].ix[testFile['longitude']<lowerLmt] = lowerLmt
testFile['longitude'].ix[testFile['longitude']>lowerLmt] = lowerLmt

matPlot.figure(figsize=(8,8))
sborn.distplot(testFile.latitude.values, bins=50, kde=False)
matPlot.xlabel('Plotting Longitude', fontsize=14)
matPlot.show()

####################So we can deduce that longitude values are mostly in the range [-73.8 , -74.02].



####################Extracting Year,Month,Day from Created###################
testFile["created"] = pread.to_datetime(testFile["created"])
testFile["createdDate"] = testFile["created"].dt.date
testFile["createdYear"]= testFile["created"].dt.year
testFile["createdMonth"]= testFile["created"].dt.month
testFile["createdDay"]= testFile["created"].dt.day

####################Converting photos and features to numeric form###################
testFile['photoCount'] = testFile['photos'].apply(len)
testFile['featureCount'] = testFile['features'].apply(len)


###################Converting description to number of words###################
testFile['numberOfWords'] = testFile['description'].apply(lambda x: len(x.split(' ')))



testFile = testFile.drop(["created","description","display_address","features","photos","street_address"], axis=1)
testFile.to_csv("resultstest.csv")