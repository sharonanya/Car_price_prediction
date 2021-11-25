# Car Price Prediction

This is my first project in Machine Learning.

# Car Price Prediction with Machine Learning

In this project we will be leveraging the power of Machine learning in predicting the price of a car without the intervention of an agent or human. This model trained will predict the price of cars based on certain factors such as the goodwill of the brand of the car, the features of the car, horsepower, mileage and many more.

## CAR PRICE PREDICTION MODEL

# Data Source: Kaggle

1.Importing Dataset and libraries:

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
data = pd.read_csv("CarPrice.csv") 
data.head()9
The dataset contains 26 columns.
2.Clean the data: Check and remove null values and to sum it up (cleaning the dataset): 

data.isnull().sum()
	This step shows that there are no null values in the dataset.
  
3.	Check the features of the data: To check the features and to get insights into the kind of data we are dealing with.

data.info() 
print(data.describe()) 
data.CarName.unique()
The price column in this dataset is the column whose values we are to predict. Let us view the distribution of the price column.
4.	Visualize the Data:     
              sns.set_style("whitegrid")
             plt.figure(figsize=(15, 10)) 
             sns.distplot(data.price) 
             plt.show()
Let us view the correlation value using a heatmap
print(data.corr()) 

plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True) 
plt.show()



Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/sharonanya/Car_price_prediction/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
