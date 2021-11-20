import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from itertools import product

# Reading the dataset and putting it into a df variable
df=pd.read_csv("online_shoppers_intention.csv")

# outputting the first 10 rows 
print("-------- PRINTING THE FIRST TEN ROWS OF THE DATAFRAME -----------")
print(df.head(10))

# Disclaimer: I added print(" ") so it looks spaced out in the console
print(" ")
# dataset column information
print("---- DATASET INFO ----")
print(df.info())

print(" ")
# description of the dataset  
print("---- DATASET DESCRIPTION ----")
print(df.describe())

print(" ")
# This sums all the null values
print("---------- SUM THE NULL ELEMENTS -----------")
print(df.isnull().sum())

print(" ")
# shows info about missing data
print("---------- MISSING DATAS -----------")
null_data = df[df.isnull().any(axis=1)]
print(null_data)

print(" ")
# Percentage of missing values.
print("---------- PERCENTAGE OF MISSING VALUES -----------")
percent_missing = (len(null_data)/len(df))*100
print(f"{percent_missing} %")

# Since all the missing data belongs to the false class(majority class),
# we can simply remove them without filling or transforming them. 
# considering we have 14 missing values each in the first 8 columns, we remove with this
df = df.dropna(axis = 0)
print(" ")

# now there's no missing values
print("---- ONCE THE MISSING DATA ARE REMOVED ----")
print(df.isnull().sum())

print(" ")
# % of the visitors generating revenue
print("--- PERCENTAGE OF VISITORS GENERATING REVENUE ---")
print(round(sum(df['Revenue'])/len(df['Revenue']), 2)*100, "% of visitors generate revenue")

print(" ")
# % of returning and new customers based on the VisitorType column in the dataset
print("--- RETURNING AND NEW CUSTOMER PERCENTAGES ---")
returning_cust_perc = round(df['VisitorType'].value_counts()['Returning_Visitor']/len(df['VisitorType'])*100, 2)
new_cust_perc = round(df['VisitorType'].value_counts()['New_Visitor']/len(df['VisitorType'])*100, 2)
print("{}% of visitors were returning".format(returning_cust_perc))
print("{}% of visitors were new".format(new_cust_perc))

# 6 different histograms for the columns stated in 'hist_columns'
hist_columns=['Administrative_Duration', 'Informational_Duration', 'ProductRelated_Duration','BounceRates', 'ExitRates', 'PageValues']
df.hist(column=hist_columns,figsize=(20,20));

# Pie chart that shows the percentage of true/false revenue
plt.figure(figsize=(10,10))
df['Revenue'].value_counts().plot(kind='pie',autopct='%1.1f', textprops={'fontsize': 15},startangle=80)
plt.title('Revenue', fontsize = 20)
plt.ylabel('')
 
# Countplot for the true and false values. 
plt.figure(figsize=(12, 8))
ax = sns.countplot(x = 'Revenue', data = df)
plt.title("No of visitors that generated revenue", fontsize=10)
plt.xlabel("Revenue Generated? (True = Yes, False = No)", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of Visitors", fontsize=16)

# this is part of the countplot. it shows the values on top of the countplots 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 
plt.show()
# This is a countplot that shows the revenue based on the different visitor types
plt.figure(figsize=(12, 8))
ax = sns.countplot(x='VisitorType', hue='Revenue' ,data = df)
plt.title("Revenue based on Visitor Type")
plt.xlabel("")
# this is part of the countplot. it shows the values on top of the countplots 
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 
plt.show()

# this is a heatmap function for each column
# this heatmap was inspired by: https://gist.github.com/drazenz/99e9a0a2b29a275170740eff0e215e4b
def heatmap(x, y, size):
    fig, ax = plt.subplots()
    plt.title("Heatmap")
    fig.set_size_inches(18.5, 10.5)
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    xval_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    yval_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    size_scale = 500
    ax.scatter(
        x=x.map(xval_to_num),
        y=y.map(yval_to_num),
        s=size * size_scale, 
        marker='s' 
    )
    # Show column labels on the axes
    ax.set_xticks([xval_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([yval_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
data=df
columns =list(df.columns) 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']

# running the heatmap code
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs()
)

plt.show()
# This is the countplot bar chart that shows the different browser values
sns.countplot(df['Browser'], palette = 'inferno')
plt.title('Browser with their count', fontsize = 20)
plt.xlabel('Browser', fontsize = 15)
plt.ylabel('count', fontsize = 15)
plt.show()

# A pie chart that shows the different percentages of visitortypes based on browsers
L=list(df['VisitorType'].value_counts().index)
s=list(df['VisitorType'].value_counts().values)
plt.rcParams['figure.figsize'] = (18, 7)
size = s
colors = ['lightgreen', 'magenta', 'blue']
labels = 'Returning_Visitor', 'New_Visitor', 'Other'
plt.pie(size, colors = colors, labels = labels, shadow = True, autopct = '%.2f%%', startangle = 0)
plt.title('Different Browsers', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()

plt.show()
# Quite a nice looking graph that shows the correlation between exit rates and revenue types (true/false)
sns.catplot(x="Revenue", y="ExitRates", hue="Weekend",
            kind="violin", data=df).set(title="Correlation between exit rates and revenue types (true/false) ")

plt.show()
# a seaborn boxenplot which shows the correlation between Administrative duration and revenue
sns.boxenplot(df['Revenue'], df['Administrative_Duration'], palette = 'pastel').set(title="correlation between Administrative duration and revenue")

plt.show()
# seaborn catplot that shows the revenue vs bouncerates based on visitor types
sns.catplot(x="Revenue", y="BounceRates", hue="VisitorType", kind="bar", data=df).set(title="revenue vs bouncerates based on visitor types")

# countplot for Month column
plt.show()
ax = sns.countplot(x="Month", data=df).set(title="Various months and their count")


# cross tab for month and revenue column
plt.show()
df1 = pd.crosstab(data['Month'], data['Revenue'])
df1.div(df1.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True, figsize = (15, 5),color=['brown','orange'], title="The months and if they produce revenue or not")

plt.show()
# a graph that shows the correlation between the different regions and the exit rates with
# the revenue
sns.violinplot(x = df['Region'], y = df['ExitRates'], hue = df['Revenue'], palette = 'spring').set(title="Exit rates and the revenues")

print("")
plt.show()
# adds the operating system value count into a size variable
size = list(df['OperatingSystems'].value_counts().values)
colors = ['orange', 'yellow', 'pink', 'crimson', 'lightgreen','blue','red','violet']
labels = "2", "1","3","4","8","6","7","5"
explode = [0, 0, 0, 0.7,1,3,4,2]
circle = plt.Circle((0, 0), 0.6, color = 'white')
# makes a pie chart that shows an exploded view of the various operating systems
plt.pie(size, colors = colors, labels = labels, explode = explode, autopct = '%.2f%%')
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.title("Blown out look of the operating systems used")
plt.legend()

# countplot for Traffic Type with revenue
plt.show()
ax = sns.countplot(x="TrafficType",hue="Revenue", data=df).set(title="Traffic Type with revenue")
plt.show()

# Datatype of each column
print("------- DATAFRAME DATATYPES ----------")
print(df.dtypes)

# Convert all categorical variables into LabelEncoder
objList=['Revenue','Month','VisitorType','Weekend']
le = LabelEncoder()
for feat in objList:
    df[feat] = le.fit_transform(df[feat])
    
print("")
# Shape of our dataset
print("------- DATAFRAME SHAPE --------")
print(df.shape)

print("")
# Our Target column is unbalanced so we need to fix it later
print("------ VALUE COUNT FOR THE REVENUE COLUMN (TARGET) --------")
print(df['Revenue'].value_counts())

# created with the help of: https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html
# Confusion matrix function for a non-normalised version
# I added the parameter "before_or_after" so i can have the text "BEFORE" or "AFTER" on my graphs
# and I wouldn't have to redefine this same function twice
def plot_confusion_matrix(cm, classes, before_or_after, model, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"Confusion matrix without normalisation {before_or_after} balancing for {model}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# Normalised confusion matrix!
def normalised_confusion_matrix_plotting(cm, classes, before_or_after, model, cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f"Confusion matrix with normalisation {before_or_after} balancing for {model}")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# axis=1 refers to columns. axis=0 would be rows
# Create X (features matrix)
X = df.drop('Revenue',axis=1).values

# Create y (labels)
y = df['Revenue'].values
    
# Splitting out data to training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Function for calculating accuracy which we use later
def accuracy(y_test,y_pred):
    outcome = confusion_matrix(y_pred,y_test)
    accur = ((outcome[0][0]+outcome[1][1])/(len(y_test)))*100
    return accur

# building a neural network
neural_n = Sequential()
neural_n.add(Dense(4, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_uniform'))
neural_n.add(Dropout(0.25))
neural_n.add(Dense(128, activation = 'relu'))
neural_n.add(Dropout(0.25))
neural_n.add(Dense( 128, activation = 'relu'))
neural_n.add(Dropout(0.25))
neural_n.add(Dense(1, activation = 'sigmoid'))
neural_n.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
neural_n.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_test,y_test) )
y_pred_n = neural_n.predict_classes(X_test)
# provides us a score for the unbalanced dataset
print(f"Accuracy Score BEFORE balancing dataset: {accuracy(y_test,y_pred_n)}")
# shows a summary of what we did above
print(neural_n.summary())

print(" ")
# these are arrays we need to store the values for the graphs which we define below
# the for loop
accuracy_data = []
nums = []
cross_val_data = []

# a for loop that checks over the random forest classifier for 10 to 100 n_estimators
print("--- EVALUATING THE MODEL (RANDOMFORESTCLASSIFIER) BEFORE DOWNSAMPLING  ---")
for i in range(10, 101, 10):
    print("Trying model with {} estimators...".format(i))
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    # this will output the model accuracy for the specified n_estimator
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100:.2f}%")
    y_model = clf.predict(X_test)
    # the below 3 will output the precision, recall and f1 score
    print(f"Model Precision score: {precision_score(y_test, y_model):.2f}")
    print(f"Model recall score: {recall_score(y_test, y_model):.2f}")
    print(f"Model f1 score: {f1_score(y_test, y_model):.2f}")
    # this will output the cross validation score
    cross_val = np.mean(cross_val_score(clf, X, y, cv=5)) * 100
    print(f"Cross-validation score: {cross_val:.2f}%")
    cross_val_data.append(cross_val)
    # this is for our graphs which we made below the for loop
    accuracy = accuracy_score(y_test, y_model) * 100
    accuracy_data.append(accuracy)
    nums.append(i)
    # this is for the AUROC code
    y_model = clf.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, y_model)
    print('Area under the ROC curve = ', round(roc_value,2))
    print(" ")
    
# graph for the Random forest classifier tests
plt.figure()
plt.plot(nums,accuracy_data)
plt.title("Random Forest Classifier estimator tests (BEFORE DOWNSAMPLING)")
plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Accuracy")
plt.show()

# graph for the Random forest classifier tests (cross validation score)
plt.plot(nums,cross_val_data)
plt.title("Random Forest Classifier estimator tests (cross validation score) (BEFORE DOWNSAMPLING)")
plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Cross Validation Score")
plt.show()

# All the models we will test before balancing the dataset
classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
    MLPClassifier(),
    LogisticRegression()
    ]

print("")
print("---------------------------------------------------------------------------")
print("---------------  TESTING DIFFERENT MODELS BEFORE BALANCING   --------------")
# This is a for loop which will output all the above models with their results
# I added this to loop over LinearSVC since it does not work with AUROC
notLinearSVC = 0
# I added this because when we print XGBClassifier, the entire parameter prints out which messes up our CM
XGBClassifier1 = 0;
for classifier in classifiers:
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier)])
    notLinearSVC += 1;
    XGBClassifier1 += 1;
    pipe.fit(X_train, y_train)   
    print(classifier)
    S=pipe.score(X_test, y_test)
    print(f'MODEL SCORE: {pipe.score(X_test, y_test)}')
    predictions = pipe.predict(X_test) 
    print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
    print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
    print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
    print(f"CLASSIFICATION REPORT FOR {classifier}")
    print(classification_report(y_test, predictions))
    # Confusion matrix where we store in a variable
    confus_matrix = confusion_matrix(y_test, predictions)
    # XGBClassifier is the 6th model in the list so we name the "model" to "XGBClassifier".
    # in the else statement, we set model to classifier. we couldnt do that for XGBClassifier because
    # the entire parameter would print out rather than just the name of the model with ()
    if (XGBClassifier1 == 6):
        # plotting the confusion matrix 
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="XGBClassifier()", before_or_after = "BEFORE")
        print("")
        # the console confusion matrix
        print("---- CONFUSION MATRIX for XGBClassifier() (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="XGBClassifier()",  before_or_after = "BEFORE")
        print("---- NORMALISED CONFUSION MATRIX for XGBClassifier() (FOR THE CONSOLE) ----")
        print(cmn)
    else:
        # plotting the confusion matrix 
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model=classifier, before_or_after = "BEFORE")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model=classifier,  before_or_after = "BEFORE")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    # looping over linearSVC
    if (notLinearSVC != 2):
        predictions = pipe.predict_proba(X_test)[:, 1]
        roc_value = roc_auc_score(y_test, predictions)
        print('Area under the ROC curve = ', round(roc_value,2))
    print("")
    print('--------------------------------------')

# even though the models' accuracy scores are quite high but the recall and f1 score is very low,
# which indicates that dataset is unbalanced.

# We fix the balancing issue with an sklearn resampling library which makes it 50/50
df_majority = df[df.Revenue==0]
df_minority = df[df.Revenue==1]
df_majority_downsampled = resample(df_majority,replace=False,n_samples= 1908, random_state=123)
 
# merge the minority (true) class with the downsampled majority (false) class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# A pie chart that shows the 50/50 balance
plt.figure(figsize=(10,10))
df_downsampled['Revenue'].value_counts().plot(kind='pie',autopct='%1.1f', textprops={'fontsize': 15},startangle=90)
plt.title('Revenue', fontsize = 20)
plt.ylabel('')

# A countplot that shows the 50/50 balance
plt.figure(figsize=(12, 8))
ax = sns.countplot(x = 'Revenue', data = df_downsampled)
plt.title("No of visitors that generated revenue AFTER BALANCING ", fontsize=10)
plt.xlabel("Revenue Generated (after downsampling)? (True = Yes, False = No)", fontsize=16)
plt.xticks(rotation=45)
plt.ylabel("Number of Visitors", fontsize=16)

# this allows for numbers to be above each plot
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}'.format(height),
            ha="center") 
plt.show()

# Display new class counts after we balance the dataset. much more balanced!
print("value count for df_downsampled.revenue is:")
print(df_downsampled.Revenue.value_counts())

# select Only best 10 column from 18 
X=df_downsampled[['Administrative', 'Administrative_Duration', 'Informational',
       'Informational_Duration', 'ProductRelated', 'ProductRelated_Duration',
       'BounceRates', 'ExitRates', 'PageValues', 'SpecialDay', 'Month',
       'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType',
       'Weekend',]]
# This replaces all the negative values with 0 values
X = X.clip(lower = 0)
# sets y to the downsampled revenue
y = df_downsampled['Revenue']

# this selects the best features
best_features = SelectKBest(score_func=chi2, k=10)
fitting = best_features.fit(X,y)
print(X)
print("")
print("----- FEATURE SCORES --------")
df_scores = pd.DataFrame(fitting.scores_)
df_columns = pd.DataFrame(X.columns)
featureScores = pd.concat([df_columns,df_scores],axis=1)
featureScores.columns = ['Specs','Score']
print(featureScores.nlargest(10,'Score'))

# Selection of Best features in our dataset
X=df_downsampled[['ProductRelated_Duration','PageValues','Administrative_Duration','Informational_Duration','ProductRelated','Administrative'
,'Informational','Month','SpecialDay','BounceRates']].values

# Second split after balancing our dataset. 70/30 training/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
print("")
# this prints out the shape of the X/y_train and X/y_test
print("X_train shape = {0}, X_test shape = {1}, y_train shape = {2}, y_test shape = {3}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))

# Function for calculating accuracy
def accuracy1(y_test,y_pred):
    outcome = confusion_matrix(y_pred,y_test)
    accur = ((outcome[0][0]+outcome[1][1])/(len(y_test)))*100
    return accur

# Neural network 
neural_n = Sequential()
neural_n.add(Dense(4, input_dim=X_train.shape[1], activation='relu', kernel_initializer='he_uniform'))
neural_n.add(Dropout(0.25))
neural_n.add(Dense(128, activation = 'relu'))
neural_n.add(Dropout(0.25))
neural_n.add(Dense( 128, activation = 'relu'))
neural_n.add(Dropout(0.25))
neural_n.add(Dense(1, activation = 'sigmoid'))
neural_n.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
neural_n.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_test,y_test) )
y_pred_n = neural_n.predict_classes(X_test)
# we use the accuracy1 function we defined above to get the accuracy score
print(f"Accuracy Score AFTER balancing dataset: {accuracy1(y_test,y_pred_n)}")
# summary of what we did above
print(neural_n.summary())

print(" ")
# We defined various arrays which we will use for the graphs we make below the for loop
accuracy_data = []
nums = []
cross_val_data = []

# As mentioned before, this is a test to test 10-100 n_estimators for randomforest
print("--- EVALUATING THE MODEL (RANDOMFORESTCLASSIFIER) BEFORE DOWNSAMPLING  ---")
for i in range(10, 101, 10):
    print("Trying model with {} estimators...".format(i))
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(X_train, y_train)
    # the model accuracy
    print(f"Model accuracy on test set: {clf.score(X_test, y_test) * 100:.2f}%")
    y_model = clf.predict(X_test)
    # the below 3 will output the various precision, recall and f1 scores
    print(f"Model Precision score: {precision_score(y_test, y_model):.2f}")
    print(f"Model recall score: {recall_score(y_test, y_model):.2f}")
    print(f"Model f1 score: {f1_score(y_test, y_model):.2f}")
    # cross validation score
    cross_val = np.mean(cross_val_score(clf, X, y, cv=5)) * 100
    print(f"Cross-validation score: {cross_val:.2f}%")
    cross_val_data.append(cross_val)
    # accuracy score which we will use for the graphs
    accuracy = accuracy_score(y_test, y_model) * 100
    accuracy_data.append(accuracy)
    nums.append(i)
    # AUROC curve scores
    y_model = clf.predict_proba(X_test)[:, 1]
    roc_value = roc_auc_score(y_test, y_model)
    print('Area under the ROC curve = ', round(roc_value,2))
    print(" ")
    
# this is the graph to show the n_estimators after balancing
plt.figure()
plt.plot(nums,accuracy_data)
plt.title("Random Forest Classifier estimator tests (AFTER DOWNSAMPLING)")
plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Accuracy")
plt.show()
# this is the graph to show the n_estimators after balancing (cross validation)
plt.plot(nums,cross_val_data)
plt.title("Random Forest Classifier estimator tests (cross validation score) (AFTER DOWNSAMPLING)")
plt.xlabel("Number of trees (n_estimators)")
plt.ylabel("Cross Validation Score")
plt.show()

# Models we will use
classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
    MLPClassifier(),
    LogisticRegression()
    ]
L=[]

print("")
print("---------------------------------------------------------------------------")
print("---------------  TESTING DIFFERENT MODELS AFTER BALANCING    --------------")
# For loop that shows scores for the above models after balancing
# I made notLinearSVC because LinearSVC doesn't work with roc_auc_score so i looped over it
notLinearSVC = 0
# we define XGBClassifier1 so we can write ' model="XGBClassifier()" ' instead of 'model = classifier'
XGBClassifier1 = 0
for classifier in classifiers:
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier)])
    notLinearSVC += 1;
    XGBClassifier1 += 1;
    pipe.fit(X_train, y_train)   
    print(classifier)
    S=pipe.score(X_test, y_test)
    # We append the various scores which we will use to find the max for the best model
    L.append(S)
    # Model scores
    print(f'MODEL SCORE: {pipe.score(X_test, y_test)}')
    predictions = pipe.predict(X_test)
    # Model precision, recall and f1 scores
    print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
    print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
    print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
    print(f"CLASSIFICATION REPORT FOR {classifier}")
    print(classification_report(y_test, predictions))
    # Confusion matrix where we store in a variable
    confus_matrix = confusion_matrix(y_test, predictions)
    # XGBClassifier is the 6th model in the list so we name the "model" to "XGBClassifier".
    # in the else statement, we set model to classifier. we couldnt do that for XGBClassifier because
    # the entire parameter would print out rather than just the name of the model with ()
    if (XGBClassifier1 == 6):
        # plotting the confusion matrix 
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="XGBClassifier()", before_or_after = "AFTER")
        print("")
        # the console confusion matrix
        print("---- CONFUSION MATRIX for XGBClassifier() (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="XGBClassifier()",  before_or_after = "AFTER")
        print("---- NORMALISED CONFUSION MATRIX for XGBClassifier() (FOR THE CONSOLE) ----")
        print(cmn)
    else:
        # plotting the confusion matrix 
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model=classifier, before_or_after = "AFTER")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model=classifier,  before_or_after = "AFTER")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    # looping over linearSVC
    if (notLinearSVC != 2):
        predictions = pipe.predict_proba(X_test)[:, 1]
        roc_value = roc_auc_score(y_test, predictions)
        print('Area under the ROC curve = ', round(roc_value,2))
    print("")
    print('--------------------------------------')

# Recall score and f1 score is improved significantly on the balanced dataset

########## IMPROVING THE MODELS #############

####Select index of Best model. L prints out all the model scores. so we set ind to the max of those scores
ind=L.index(max(L))

# this is the Grid method which has the best parameters for each model
# it uses the ind value to select the best model and use that to tune our model
def Grid(ind):
    # for example, if ind is set to 0, it will mean KNeighbourClassifier is the best so it will tune the model
    if ind==0:
        param_grid = dict(n_neighbors=[1,2,3,4,5,6,7,8])
        c1_B = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,cv=5)
        c1_B.fit(X_train, y_train)
        print(c1_B,c1_B.best_params_)
        return c1_B.best_params_

    elif ind==1:
        c2_B = GridSearchCV(LinearSVC(),param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
        c2_B.fit(X_train,y_train)
        print(c2_B,c2_B.best_params_)
        return c2_B.best_params_

    elif ind==2:
        param_grid = {
            'criterion':['gini','entropy'],
            'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
            }
        c3_B = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5)
        c3_B.fit(X_train,y_train)
        print(c3_B,c3_B.best_params_)
        return c3_B.best_params_

    elif ind==3:
        param_grid={
            'n_estimators': [2,200], 
            'max_depth': [None, 10, 20, 30], 
            'min_samples_split': [1, 2, 3],
            'min_samples_leaf':[1,2,3]
            }
        c4_B = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
        c4_B.fit(X_train,y_train)
        print(c4_B,c4_B.best_params_)
        return c4_B.best_params_

    elif ind==4:
        param_grid={
            'n_estimators':[100,150,200],
            'learning_rate':[.001,0.01,.1]
            }
        c5_B = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid, cv= 5)
        c5_B.fit(X_train,y_train)
        print(c5_B,c5_B.best_params_)
        return c5_B.best_params_

    elif ind==5:
        param_grid={
            'max_depth': range (2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.01, 0.05]
        }
        c6_B = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv= 5)
        c6_B.fit(X_train,y_train)
        print(c6_B,c6_B.best_params_)
        return c6_B.best_params_

    elif ind==6:
        parameters = {
            'solver': ['lbfgs'], 
            'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000], 
            'alpha': 10.0 ** -np.arange(1, 10), 
            'hidden_layer_sizes':np.arange(10, 15), 
            'random_state':[0,1,2,3,4,5,6,7,8,9]
            }
        c7_B = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters, cv= 5)
        c7_B.fit(X_train,y_train)
        print(c7_B,c7_B.best_params_)
        return c7_B.best_params_

    elif ind==7:
        parameters = {
            'penalty': ['l1','l2'],
            'C': [0.001,0.01,0.1,1,10,100,1000]
            }
        c8_B = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv= 5)
        c8_B.fit(X_train,y_train)
        print(c8_B,c8_B.best_params_)
        return c8_B.best_params_

print("----- Apply Cross validation on Best model and find out best parameters of our selected model -------")
# this applies cross validation
para=Grid(ind)
print(para)

# outputs the score depending on the best model
def re_run(ind, para,X_train,Y_train,X_test,Y_test):
    if ind==0:
        c=KNeighborsClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"KNeighbour Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"KNeighbour Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test)
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("KNeighbour Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="KNeighborsClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="KNeighborsClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
        
    elif ind==1:
        c=LinearSVC(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"LinearSVC Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"LinearSVC Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("LinearSVC Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="LinearSVC", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="LinearSVC",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==2:
        c=DecisionTreeClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"DecisionTreeClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"DecisionTreeClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("DecisionTreeClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="DecisionTreeClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="DecisionTreeClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==3:
        c=RandomForestClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"RandomForestClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"RandomForestClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("RandomForestClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="RandomForestClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="RandomForestClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==4:
        c=AdaBoostClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"AdaBoostClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"AdaBoostClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("AdaBoostClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="AdaBoostClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="AdaBoostClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==5:
        c=XGBClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"XGBClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"XGBClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("XGBClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="XGBClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="XGBClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==6:
        c=MLPClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"MLPClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"MLPClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("MLPClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="MLPClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="MLPClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==7:
        c=LogisticRegression(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"LogisticRegression Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"LogisticRegression Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("LogisticRegression Classification Report:")
        print(classification_report(Y_test, predictions))
          # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="LogisticRegression", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="LogisticRegression",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    
print("")
print("------ RUNNING THE BEST MODEL WITH THE BEST PARAMETERS ------")
# running the best model
re_run(ind, para,X_train,y_train,X_test,y_test)

# now we will run the second best test without the best model
classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(),
    XGBClassifier(),
    MLPClassifier(),
    LogisticRegression()
    ]
L=[]

print("")
print("----------------------------------------------------------------------------------------")
print("---------------  LOOKING AT SECOND BEST MODEL (WITHOUT THE PREVIOUS BEST) --------------")
# loops over it
notLinearSVC = 0
for classifier in classifiers:
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier)])
    notLinearSVC += 1;
    pipe.fit(X_train, y_train)   
    print(classifier)
    S=pipe.score(X_test, y_test)
    # L will add the accuracy scores of each model
    L.append(S)
    # printing the model score
    print(f'MODEL SCORE: {pipe.score(X_test, y_test)}')
    predictions = pipe.predict(X_test) 
    # printing the precision, recall and f1 scores
    print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
    print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
    print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
    print(f"CLASSIFICATION REPORT FOR {classifier}")
    # classification report
    print(classification_report(y_test, predictions))

    # looping over linearSVC
    if (notLinearSVC != 2):
        predictions = pipe.predict_proba(X_test)[:, 1]
        roc_value = roc_auc_score(y_test, predictions)
        print('Area under the ROC curve = ', round(roc_value,2))
    print("")
    print('--------------------------------------')

# selecting the best model
ind=L.index(max(L))

# a grid method to use the best parameter for tuning
def Grid(ind):
    if ind==0:
        param_grid = dict(n_neighbors=[1,2,3,4,5,6,7,8])
        c1_B = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,cv=5)
        c1_B.fit(X_train, y_train)
        print(c1_B,c1_B.best_params_)
        return c1_B.best_params_

    elif ind==1:
        c2_B = GridSearchCV(LinearSVC(),param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
        c2_B.fit(X_train,y_train)
        print(c2_B,c2_B.best_params_)
        return c2_B.best_params_

    elif ind==2:
        param_grid = {
            'criterion':['gini','entropy'],
            'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
            }
        c3_B = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5)
        c3_B.fit(X_train,y_train)
        print(c3_B,c3_B.best_params_)
        return c3_B.best_params_

    elif ind==3:
        param_grid={
            'n_estimators':[100,150,200],
            'learning_rate':[.001,0.01,.1]
            }
        c4_B = GridSearchCV(estimator=AdaBoostClassifier(), param_grid=param_grid, cv= 5)
        c4_B.fit(X_train,y_train)
        print(c4_B,c4_B.best_params_)
        return c4_B.best_params_

    elif ind==4:
        param_grid={
            'max_depth': range (2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.01, 0.05]
        }
        c5_B = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv= 5)
        c5_B.fit(X_train,y_train)
        print(c5_B,c5_B.best_params_)
        return c5_B.best_params_

    elif ind==5:
        parameters = {
            'solver': ['lbfgs'], 
            'max_iter': [1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000 ], 
            'alpha': 10.0 ** -np.arange(1, 10), 
            'hidden_layer_sizes':np.arange(10, 15), 
            'random_state':[0,1,2,3,4,5,6,7,8,9]
            }
        c6_B = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters, cv= 5)
        c6_B.fit(X_train,y_train)
        print(c6_B,c6_B.best_params_)
        return c6_B.best_params_

    elif ind==6:
        parameters = {
            'penalty': ['l1','l2'],
            'C': [0.001,0.01,0.1,1,10,100,1000]
            }
        c7_B = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv= 5)
        c7_B.fit(X_train,y_train)
        print(c7_B,c7_B.best_params_)
        return c7_B.best_params_

print("----- Apply Cross validation on SECOND BEST model and find out best parameters of our selected model -------")
# second best test
para=Grid(ind)
print(para)
def re_run(ind, para,X_train,Y_train,X_test,Y_test):
    if ind==0:
        c=KNeighborsClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"KNeighbour Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"KNeighbour Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test)
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("KNeighbour Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="KNeighborsClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="KNeighborsClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==1:
        c=LinearSVC(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"LinearSVC Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"LinearSVC Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("LinearSVC Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="LinearSVC", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="LinearSVC",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==2:
        c=DecisionTreeClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"DecisionTreeClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"DecisionTreeClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("DecisionTreeClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="DecisionTreeClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="DecisionTreeClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==3:
        c=AdaBoostClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"AdaBoostClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"AdaBoostClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("AdaBoostClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="AdaBoostClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="AdaBoostClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==4:
        c=XGBClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"XGBClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"XGBClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("XGBClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="XGBClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="XGBClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==5:
        c=MLPClassifier(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"MLPClassifier Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"MLPClassifier Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("MLPClassifier Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="MLPClassifier", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="MLPClassifier",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    elif ind==6:
        c=LogisticRegression(**para)
        c.fit(X_train,Y_train)
        print(c)
        print(f"LogisticRegression Score (X_train, Y_train) = {c.score(X_train, Y_train)}")
        print(f"LogisticRegression Score (X_test, Y_test) = {c.score(X_test, Y_test)}")
        predictions = c.predict(X_test) 
        print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
        print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
        print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
        print("LogisticRegression Classification Report:")
        print(classification_report(Y_test, predictions))
        # plotting the confusion matrix 
        confus_matrix = confusion_matrix(y_test, predictions)
        plt.show()
        plot_confusion_matrix(confus_matrix, classes=['False','True'], model="LogisticRegression", before_or_after = "AFTER TUNING")
        print("")
        # the console confusion matrix
        print(f"---- CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(confus_matrix)
        print("")
        # this is for our normalised confusion matrix
        cmn = confus_matrix.astype('float') / confus_matrix.sum(axis=1)[:, np.newaxis]
        # plotting the normalised confusion matrix 
        plt.show()
        normalised_confusion_matrix_plotting(confus_matrix, classes=['False','True'], model="LogisticRegression",  before_or_after = "AFTER TUNING")
        print(f"---- NORMALISED CONFUSION MATRIX for {classifier} (FOR THE CONSOLE) ----")
        print(cmn)
    
print("")
print("------ RUNNING THE BEST MODEL WITH THE BEST PARAMETERS ------")
# running the second best test
re_run(ind, para,X_train,y_train,X_test,y_test)

# Running tests for the rest of the models
classifiers = [
    KNeighborsClassifier(),
    LinearSVC(),
    DecisionTreeClassifier(),
    XGBClassifier(),
    MLPClassifier(),
    LogisticRegression()
    ]
L=[]

# loops over it
print("")
print("---------------------------------------------------")
print("----------- NOW RUNNING FOR THE REST --------------")
for classifier in classifiers:
    pipe = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', classifier)])
    pipe.fit(X_train, y_train)   
    print(classifier)
    S=pipe.score(X_test, y_test)
    L.append(S)
    print(f'MODEL SCORE: {pipe.score(X_test, y_test)}')
    predictions = pipe.predict(X_test) 
    print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
    print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
    print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
    print(f"CLASSIFICATION REPORT FOR {classifier}")
    print(classification_report(y_test, predictions))

    print("")
    print('--------------------------------------')

# # a grid method to use the best parameter for tuning
def Grid(ind):
    if ind==0:
        param_grid = dict(n_neighbors=[1,2,3,4,5,6,7,8])
        c1_B = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid,cv=5)
        c1_B.fit(X_train, y_train)
        print(c1_B,c1_B.best_params_)
        return c1_B.best_params_
    
    elif ind==1:
        c2_B = GridSearchCV(LinearSVC(),param_grid={"C": [0.001, 0.01, 0.1, 1.0, 10.0]})
        c2_B.fit(X_train,y_train)
        print(c2_B,c2_B.best_params_)
        return c2_B.best_params_

    elif ind==2:
        param_grid = {
            'criterion':['gini','entropy'],
            'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]
            }
        c3_B = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, cv=5)
        c3_B.fit(X_train,y_train)
        print(c3_B,c3_B.best_params_)
        return c3_B.best_params_

    elif ind==3:
        param_grid={
            'max_depth': range (2, 10, 1),
            'n_estimators': range(60, 220, 40),
            'learning_rate': [0.1, 0.01, 0.05]
        }
        c5_B = GridSearchCV(estimator=XGBClassifier(), param_grid=param_grid, cv= 5)
        c5_B.fit(X_train,y_train)
        print(c5_B,c5_B.best_params_)
        return c5_B.best_params_

    elif ind==4:
        parameters = {
            'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.05],
            'learning_rate': ['constant','adaptive'],
            }
        c6_B = GridSearchCV(estimator=MLPClassifier(), param_grid=parameters, cv= 5)
        c6_B.fit(X_train,y_train)
        print(c6_B,c6_B.best_params_)
        return c6_B.best_params_

    elif ind==5:
        parameters = {
            'penalty': ['l1','l2'],
            'C': [0.001,0.01,0.1,1,10,100,1000]
            }
        c7_B = GridSearchCV(estimator=LogisticRegression(), param_grid=parameters, cv= 5)
        c7_B.fit(X_train,y_train)
        print(c7_B,c7_B.best_params_)
        return c7_B.best_params_

# Over here, we are attempting to tune the rest of the models with our function Grid()
print("")
print("-------------------------------------------------------------")
print("RUNNING TUNING FOR THE REST OF THE MODELS")
print("-------------------------------------------------------------")

para=Grid(0)
print(para)
c=KNeighborsClassifier(**para)
c.fit(X_train,y_train)
print(c)
print(f"KNeighbour Score (X_train, y_train) = {c.score(X_train, y_train)}")
print(f"KNeighbour Score (X_test, y_test) = {c.score(X_test, y_test)}")
predictions = c.predict(X_test)
print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
print("-------------------------------------------------------------")

para=Grid(1)
print(para)
c=LinearSVC(**para)
c.fit(X_train,y_train)
print(c)
print(f"LinearSVC Score (X_train, y_train) = {c.score(X_train, y_train)}")
print(f"LinearSVC Score (X_test, y_test) = {c.score(X_test, y_test)}")
predictions = c.predict(X_test)
print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
print("-------------------------------------------------------------")

para=Grid(2)
print(para)
c=DecisionTreeClassifier(**para)
c.fit(X_train,y_train)
print(c)
print(f"DecisionTreeClassifier Score (X_train, y_train) = {c.score(X_train, y_train)}")
print(f"DecisionTreeClassifier Score (X_test, y_test) = {c.score(X_test, y_test)}")
predictions = c.predict(X_test)
print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
print("-------------------------------------------------------------")

para=Grid(3)
print(para)
c=XGBClassifier(**para)
c.fit(X_train,y_train)
print(c)
print(f"XGBClassifier Score (X_train, y_train) = {c.score(X_train, y_train)}")
print(f"XGBClassifier Score (X_test, y_test) = {c.score(X_test, y_test)}")
predictions = c.predict(X_test)
print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
print("-------------------------------------------------------------")

# MLPClassifier takes quite a long time to run so please be patient
para=Grid(4)
print(para)
c=MLPClassifier(**para)
c.fit(X_train,y_train)
print(c)
print(f"MLPClassifier Score (X_train, y_train) = {c.score(X_train, y_train)}")
print(f"MLPClassifier Score (X_test, y_test) = {c.score(X_test, y_test)}")
predictions = c.predict(X_test)
print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")
print("-------------------------------------------------------------")

para=Grid(5)
print(para)
c=LogisticRegression(**para)
c.fit(X_train,y_train)
print(c)
print(f"LogisticRegression Score (X_train, y_train) = {c.score(X_train, y_train)}")
print(f"LogisticRegression Score (X_test, y_test) = {c.score(X_test, y_test)}")
predictions = c.predict(X_test)
print(f"Model Precision score: {precision_score(y_test, predictions):.2f}")
print(f"Model recall score: {recall_score(y_test, predictions):.2f}")
print(f"Model f1 score: {f1_score(y_test, predictions):.2f}")