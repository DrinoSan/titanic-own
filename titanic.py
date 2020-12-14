# imports of all the libaries i need
import re
import pandas as pd
import numpy as np
import random as rnd

# Algorithm
from sklearn.ensemble import RandomForestClassifier
# RandomForestClassiefier seems to work the best


print("Start")
# The basic data exploration and visualization has been done in the jupyter notebook
# Therefore i will not repeat the steps in this


test_df = pd.read_csv("../Jupyter/titanic/test.csv")
train_df = pd.read_csv("../Jupyter/titanic/train.csv")


data = [train_df, test_df]
for dataset in data:
    # A new column is genereated with the name "relatives" it is the result of the addition of "SibSp" and "Parch"
    dataset["relatives"] = dataset["SibSp"] + dataset["Parch"]
    # loc generates the column if it is not existing at the beginning
    # With loc we iterate over every row and ask for a condition -> if the value in dataset["relatives"] > 0 we wil write 0
    # into the "not_alone" column. Or 1 if the value is equal 1.(This means the person IS ALONE)
    dataset.loc[dataset["relatives"] > 0, "not_alone"] = 0
    dataset.loc[dataset["relatives"] == 0, "not_alone"] = 1
    dataset["not_alone"] = dataset["not_alone"].astype(int)


# ---------------------------------
# In the next Step we will try to convert the "Embarked" column into a numeric column
# because it seems that there is a correaltion because "Embarked" indicates on which floor the person
# has had his room. This probably seems to be interesting

# We will import the re libary for regex expressions
deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8}
data = [train_df, test_df]


for dataset in data:
    dataset["Cabin"].fillna("U0", inplace=True)
    for i in dataset.index:
        dataset.at[i, "Cabin"] = dataset["Cabin"][i][:1]
        dataset["Deck"] = dataset["Cabin"].map(deck)
    dataset["Deck"] = dataset["Deck"].fillna(0)
    dataset["Deck"] = dataset["Deck"].astype(int)

train_df = train_df.drop(["Cabin"], axis=1)
test_df = test_df.drop(["Cabin"], axis=1)


# Counting most frequent value in Embarked
# train_df["Embarked"].value_counts().idxmax()

data = [train_df, test_df]
for dataset in data:
    dataset["Embarked"].fillna(
        dataset["Embarked"].value_counts().idxmax(), inplace=True)


data = [train_df, test_df]
for dataset in data:
    pclass1_mean = dataset[dataset["Pclass"] == 1]["Age"].mean().round(1)
    pclass2_mean = dataset[dataset["Pclass"] == 2]["Age"].mean().round(1)
    pclass3_mean = dataset[dataset["Pclass"] == 3]["Age"].mean().round(1)

    dataset.loc[(dataset["Age"].isnull() == True) & (
        dataset["Pclass"] == 1), ["Age"]] = pclass1_mean
    dataset.loc[(dataset["Age"].isnull() == True) & (
        dataset["Pclass"] == 2), ["Age"]] = pclass2_mean
    dataset.loc[(dataset["Age"].isnull() == True) & (
        dataset["Pclass"] == 3), ["Age"]] = pclass3_mean

    # just to check
    dataset["Age"].isna().sum()


# New Features
data = [train_df, test_df]
for dataset in data:
    dataset["Age_Class"] = dataset["Age"] * dataset["Pclass"]

# Fare per Person
data = [train_df, test_df]
for dataset in data:
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
###########

# Need to redo this part i dont like the way it is done
data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in data:
    # extract titles
    dataset["Title"] = dataset["Name"].str.extract(
        "([A-Za-z]+)\.", expand=False)
    # replace titles wiht a more commong tilte or as Rare
    dataset["Title"] = dataset["Title"].replace(["Lady", "Countess", "Capt", "Col", "Don", "Dr",
                                                 "Major", "Rev", "Sir", "Jonkheer", "Dona"], "Rare")
    dataset["Title"] = dataset["Title"].replace("Mlle", "Miss")
    dataset["Title"] = dataset["Title"].replace("Ms", "Miss")
    dataset["Title"] = dataset["Title"].replace("Mme", "Mrs")
    dataset["Title"] = dataset["Title"].map(titles)
    # filling Nan with 0, to get safe
    dataset["Title"] = dataset["Title"].fillna(0)


train_df = train_df.drop(["Name"], axis=1)
test_df = test_df.drop(["Name"], axis=1)

train_df = pd.get_dummies(
    train_df, columns=["Embarked", "Sex", "Pclass"], drop_first=True)
test_df = pd.get_dummies(
    test_df, columns=["Embarked", "Sex", "Pclass"], drop_first=True)


data = [train_df, test_df]
for dataset in data:
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean().round(1))
    dataset["Fare"] = dataset["Fare"].astype(int)


# The groups were produced by the qcut function
#categories = pd.qcut(train_df["Fare"], q=6)

data = [train_df, test_df]
for dataset in data:
    dataset.loc[dataset['Fare'] <= 7, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7) & (
        dataset['Fare'] <= 14), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14) & (
        dataset['Fare'] <= 26), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 26.0) & (
        dataset['Fare'] <= 52), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 52) & (
        dataset['Fare'] <= 512), 'Fare'] = 4
    dataset.loc[dataset['Fare'] > 512, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(["Ticket"], axis=1)
test_df = test_df.drop(["Ticket"], axis=1)


train_df = train_df.drop("PassengerId", axis=1)

train_df = train_df.drop("Embarked_Q", axis=1)
test_df = test_df.drop("Embarked_Q", axis=1)

train_df = train_df.drop("Pclass_2", axis=1)
test_df = test_df.drop("Pclass_2", axis=1)

train_df = train_df.drop("not_alone", axis=1)
test_df = test_df.drop("not_alone", axis=1)

train_df = train_df.drop("Parch", axis=1)
test_df = test_df.drop("Parch", axis=1)

train_df = train_df.drop("Embarked_S", axis=1)
test_df = test_df.drop("Embarked_S", axis=1)

########
# Now we can start to train the model
########

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# I will stick to the RandomForestClassifier because this method has shown the best results of all

model = RandomForestClassifier(n_estimators=100, max_depth=2)
model.fit(X_train, Y_train)

Y_prediction = model.predict(X_test)
########


###########
# Ausgabe f√ºr Kaggle
output = pd.DataFrame(
    {'PassengerId': test_df.PassengerId, 'Survived': Y_prediction})
output.to_csv('my_submission.csv', index=False)
print('Submission successfully saved')

# Should still check the pclass stuff and embarked aswell as the cabin stuff
# I deleted the categories because the cabin goes from 1 to 9 meaning 1 was a cabin near the ground and 9 was higher
# this is why the categories dont seem very smart....
