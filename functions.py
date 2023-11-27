
#Import the required Libraries
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.ensemble import AdaBoostClassifier

#Initialize each algorithm
NBModel= GaussianNB()
modelQDN = QuadraticDiscriminantAnalysis()
DTCModel = DecisionTreeClassifier()
adaboost = AdaBoostClassifier()
mlp = MLPClassifier()
KNC_Model = KNeighborsClassifier()
linearRegModel = LogisticRegression()
percepModel = Perceptron()
randomForest = RandomForestClassifier()

st.set_page_config(layout="wide")

# Functions for each of the pages
def home(dataset):
    if dataset:
        st.header('Explore Dataset')
    else:
        st.header('Dataset Preview')

def data_summary():
    st.header('Statistics of  Dataframe')
    st.write(dataset.describe())


def data_header():
    st.header('Header of Dataframe')
    st.write(dataset.head())


def plotFunc(dataset):
    labels=dataset.Label
    plotData= dataset.value_counts(labels)
    st.subheader("Count of each Attacks")
    st.bar_chart(plotData)
    
    
def evaluate_algorithm(algo_name, clf):
    X =dataset.drop(columns=['Label'])
    y =dataset['Label']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1, start_time, end_time

def weightRanker(csv_file, target_column, num_features):

    # Separate features (X) and target (y)
    X = csv_file.drop(target_column, axis=1)
    y = csv_file[target_column]

    # Use SelectKBest with ANOVA F-test to compute feature scores
    selector = SelectKBest(score_func=f_classif, k=num_features)
    selector.fit(X, y)

    # Get the feature scores
    feature_scores = selector.scores_

    # Create a DataFrame to store feature names and their scores
    feature_ranking = pd.DataFrame({'Feature': X.columns, 'Score': feature_scores})

    # Sort features by score in descending order
    feature_ranking = feature_ranking.sort_values(by='Score', ascending=True)

    return feature_ranking
