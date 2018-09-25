import numpy as np 
from sklearn import cross_validation, metrics
from sklearn import svm 
from sklearn.preprocessing import MinMaxScaler
import pickle 
import sqlite3
import random as rd 
import matplotlib.pyplot as plt 
from matplotlib import style 
import time

# File path for CSV file with data
crypto_file_path = "crypto.csv"

def get_counter(dataset):
    """
    Counts the number of occurences for each label
    """
    label_collection = [i[-1] for i in dataset]
    label_set = set(label_collection)
    counter = {}
    for label in label_set:
        counter[label] = 0
    for label in label_collection:
        counter[label] += 1
    return counter 

def get_keys(counter):
    """
    Returns keys from the counter dictionary
    """
    return(
        [key for key in counter]
    )

def get_values(counter):
    """
    Returns the counter values from the key value pair
    """
    return(
        [counter[key] for key in 
            get_keys(counter)
        ]
    )

def get_lowest_frequency(counter):
    """
    Returns the lowest value from get_values()
    """
    return min(
        get_values(counter)
    )

def get_datapoints_by_label(dataset, counter):
    """
    Returns all the datapoints with a given label
    """
    different_labels = get_keys(counter)
    datapoints_by_label = []
    for label in different_labels:
        temp_array = []
        for datapoint in dataset:            
            if float(datapoint[-1]) == float(label):                
                temp_array.append(datapoint)
        datapoints_by_label.append(temp_array)
    return datapoints_by_label

def rebalance(dataset, counter):
    """
    Obtains the label with the lowest frequency and creates a dataset
    that has this same amount of occurences for each label
    """
    # Get label with lowest frequency
    lowest_frequency = get_lowest_frequency(counter)
    print("\nObtaining datapoints so that there are always {} datapoints for each label\n".format(lowest_frequency))
    # Get list of datapoints by label
    datapoints_by_label = get_datapoints_by_label(dataset, counter)
    # Get only the number of datapoints per label as the lowest frequency
    rebalanced_data = []
    for d in datapoints_by_label:
        d = d[:lowest_frequency]
        rebalanced_data.extend(d)
    return rebalanced_data

def get_coins_and_prices(file_name):
    """
    Retrieves all coins with their correspondent price
    array and stores it as a dictionary
    """
    print("\nRetrieving coin data from {}".format(file_name))
    # Stores a list of each coin name
    coins = []
    # A dictionary where:
    # Key = Coin name
    # Value = List of prices
    coins_and_prices = {}
    with open(file_name, 'r') as file:
        # Remove first line because it has the table headings
        first_line = file.readline()                
        for line in file:
            item = line.split(",")
            coin = str(item[2])
            if coin not in coins:            
                coins.append(coin)
                coins_and_prices[coin] = []
            price = float(item[8])
            coins_and_prices[coin].append(price)            
    print("\nAll coin data retrieved")
    return coins_and_prices

def binary(input):
    """
    Converts a price change to binary:
    --> If the price increase, return 1
    --> If the price decreased, return 0
    """
    if input > 0:
        return 1
    return 0

def create_data_set(assets_and_prices, window):
    """
    Creates a dataset where the elements of the array are changes
    in price from one day to the next and the label is a binary number,
    describing if the price increased or decreased    
    """
    # Increment window by one so that the last element of the list
    # can be used as the label
    window += 1
    dataset = []
    for key in assets_and_prices:
        prices = assets_and_prices[key]
        # Create window frames of prices
        window_frames = [
            prices[x:x+window] for x in range(0, len(prices), window)
        ]    
        # Exclude those that do not have the length of the window
        window_frames = [
            i for i in window_frames if len(i) == window 
        ]
        for window_frame in window_frames:
            datapoint = []
            for i in range(len(window_frame)-1):
                x = window_frame[i]
                y = window_frame[i+1]
                # If value is zero, change it to a value close to zero
                if x == 0:
                    x = 0.0001
                if y == 0:
                    y = 0.0001                
                # Append change in price
                datapoint.append(
                    float(((y-x)/x)*100)
                )             
            dataset.append(datapoint)
        # Convert labels to binary
        for d in dataset:
            d[-1] = binary(d[-1])
    return dataset
    
def get_features_and_labels(dataset):
    """
    Extracts features and labels from the dataset and
    converts them to np arrays
    """
    features = []
    labels = []
    for d in dataset:
        features.append(d[:-1])
        labels.append(d[-1])        
    return np.array(features), np.array(labels)

def create_classifier(C, gamma):
    """
    Returns an untrained SVM classifier
    with the C and gamma from the method
    inputs
    """
    classifier = svm.SVC(C=C, gamma=gamma)
    return classifier 

def fitness_value(clf, X_train, y_train, X_test, y_test):        
    """
    Calculates the fitness value of the model,
    which is the accuracy
    """
    fitness_value = 0
    try:
        clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted)        
        fitness_value = accuracy * 100 
    except ValueError:
        fitness_value = 50 
    return fitness_value


class Combination:
    
    """
    A class whose objects store the different combinations
    created throughout the simulated annealing process
    """

    def __init__(self, x, y, accuracy):
        self.x = x 
        self.y = y 
        self.accuracy = accuracy


def get_optimal_model(X_train, X_test, y_train, y_test):

    """
    This is the simulated annealing process.
    The method returns an untrained SVM classifier
    with the C and gamma hyperparameters selected
    through simulated annealing
    """

    best_combinations = []

    # Random initial solution we would like to start at
    print("\nGenerating random C and gamma...")
    C = rd.randint(100,1000)
    gamma = float((rd.randint(1,10)))   
    print("C = {} and gamma = {}".format(C, gamma))
    time.sleep(5)
    initial_combination = Combination(C, gamma, 0)

    best_combinations.append(initial_combination)

    k = 0.8 # Movement - we multiply a random number by k and add it to get the new x to move in the search space
    T0 = 1000 # Initial temperature
    M = 300
    N = 100
    alpha = 0.85 # Cooling rate

    # Main nested loop
    print("Progress: ")
    for i in range(M):
                
        print("\tM = {}/{}".format(i,M))

        for j in range(N):
            
            # Obtain last values from the best combinations list
            temp_C = best_combinations[-1].x  
            temp_gamma = best_combinations[-1].y 

            # Should we add or subtract
            # (k * the next random number)?
            ran_x_1 = np.random.rand()
            ran_x_2 = np.random.rand()

            # Same as for ran_x_1
            ran_y_1 = np.random.rand()
            ran_y_2 = np.random.rand()

            x1 = 0
            y1 = 0

            if ran_x_1 >= 0.5:
                x1 = k * ran_x_2
            else:
                x1 = -k * ran_x_2

            if ran_y_1 >= 0.5:
                y1 = k * ran_y_2
            else:
                y1 = -k * ran_y_2

            temp_C = C + (x1*100)  
            temp_gamma = gamma + y1 

            # Ensure that hyperparameters are within a range
            while ((temp_C < 10) or (temp_C > 1000)) and ((temp_gamma < 1) or (temp_gamma > 10)):
            
                # Should we add or subtract
                # (k * the next random number)?
                ran_x_1 = np.random.rand()
                ran_x_2 = np.random.rand()

                # Same as for ran_x_1
                ran_y_1 = np.random.rand()
                ran_y_2 = np.random.rand()

                x1 = 0
                y1 = 0

                if ran_x_1 >= 0.5:
                    x1 = k * ran_x_2
                else:
                    x1 = -k * ran_x_2

                if ran_y_1 >= 0.5:
                    y1 = k * ran_y_2
                else:
                    y1 = -k * ran_y_2

                temp_C = C + (x1*100)
                temp_gamma = gamma + y1 
            
            # Objective function of current combination
            of_current = fitness_value(create_classifier(C, gamma), X_train, y_train, X_test, y_test)

            # Objective function of new combination                     
            of_new = fitness_value(create_classifier(temp_C, temp_gamma), X_train, y_train, X_test, y_test)
                                    
            ran_1 = np.random.rand()
            
            formula = 1 / (np.exp((of_new-of_current)/T0))

            if of_new > of_current:
                # If this is true - the temporary solution becomes the new solution
                C = temp_C
                gamma = temp_gamma                     
            # If the new solution is actually worse than the current
            # we check with the random numeber to see if to take
            # the leap and try the next solution
            elif ran_1 <= formula:
                C = temp_C
                gamma = temp_gamma
                        
            # Get the fitness value from an untrained classifier with the best hyperparameters
            fv = fitness_value(create_classifier(C, gamma), X_train, y_train, X_test, y_test)    

            # If this fitness value is better than the previous one
            # then set it as the new best combination
            if fv > best_combinations[-1].accuracy:
                best_combinations[-1] = Combination(C, gamma, fv)

                                        
        """ After this is done n times """
        T0 = alpha * T0         
        # Append the best combination to the list of best combinations
        best_combinations.append(best_combinations[-1])

                                    
    """ Plot the graph """    
    x = [i for i in range(len(best_combinations))]
    y = [float(i.accuracy) for i in best_combinations]
    plt.plot(x,y)
    plt.title("Fitness value over time", fontsize=20, fontweight='bold')
    plt.xlabel("Stage", fontsize=15, fontweight='bold')
    plt.ylabel("Fitness value", fontsize=15, fontweight='bold')
    plt.show() 

    # Sort the best combinations list to get the top best combination
    best_combination = sorted(best_combinations, key = lambda c : c.accuracy, reverse=True)[0]
    
    """ Returned the untrained classifier """               
    classifier = create_classifier(
        best_combination.x, best_combination.y 
    )

    print("BEST COMBINATION: C = {} and gamma = {}".format(
        best_combination.x,
        best_combination.y
    ))

    return classifier


def get_best_model(dataset):
    """
    Prepares the dataset, trains the best model
    with the data and returns it
    """
    best_model_accuracy = 0  
    best_model = 0
    # Shuffle Dataset
    rd.shuffle(dataset)   
    # Create dataset
    features = []
    for i in dataset:
        i = i[:-1]
        features.append(i)                        
    labels = []
    for i in dataset:
        labels.append(i[-1])
    # Transform
    scaler = MinMaxScaler()
    features = list(scaler.fit_transform(features))
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.3)                    
    # Running simulated annealing
    best_model = get_optimal_model(X_train, X_test, y_train, y_test) 
    best_model.fit(X_train, y_train)
    predicted = best_model.predict(X_test)
    best_model_accuracy = metrics.accuracy_score(y_test, predicted) * 100
    # *************************   
    return best_model, best_model_accuracy

# This is the lookback window for the price movements,
# in other words, the number of price changes from day to day
window = 8

# Create dataset
coins_and_prices = get_coins_and_prices(crypto_file_path)
dataset = create_data_set(coins_and_prices, window)

# Rebalance Dataset
dataset = rebalance(dataset, get_counter(dataset))

dataset_size = len(dataset)
print("\nNumber of datapoints = {}".format(dataset_size))

# Get %0.1 of the dataset
rd.shuffle(dataset)
proportion = int((dataset_size/100)*0.1)
dataset = dataset[:proportion]

best_model, best_model_accuracy = get_best_model(dataset)

print("Accuracy achieved = {}%".format(round(best_model_accuracy,2)))