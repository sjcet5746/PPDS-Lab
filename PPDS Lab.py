#!/usr/bin/env python
# coding: utf-8

# In[1]:


#1. Write a program to demonstrate a) Different numeric data types and b) To perform different 
#Arithmetic Operations on numbers in Python
# Different numeric data types
integer_num = 10
float_num = 3.14
complex_num = 2 + 3j

# Arithmetic operations
addition = integer_num + float_num
subtraction = float_num - integer_num
multiplication = integer_num * float_num
division = float_num / integer_num
exponentiation = integer_num ** 2
modulus = integer_num % 3
integer_division = integer_num // 3

# Displaying results
print("Different Numeric Data Types:")
print("Integer Number:", integer_num)
print("Float Number:", float_num)
print("Complex Number:", complex_num)

print("\nArithmetic Operations:")
print("Addition:", addition)
print("Subtraction:", subtraction)
print("Multiplication:", multiplication)
print("Division:", division)
print("Exponentiation:", exponentiation)
print("Modulus:", modulus)
print("Integer Division:", integer_division)


# In[2]:


#2. Write a program to create, append, and remove lists in Python
# Create an empty list
my_list = []

# Append elements to the list
my_list.append(1)
my_list.append(2)
my_list.append(3)
print("List after appending elements:", my_list)

# Remove elements from the list
my_list.remove(2)  # Remove the element with value 2
print("List after removing element 2:", my_list)

# Create a new list
new_list = [4, 5, 6]

# Extend the original list by adding elements from the new list
my_list.extend(new_list)
print("List after extending with new_list:", my_list)

# Remove the first occurrence of a specific value from the list
my_list.remove(4)
print("List after removing element 4:", my_list)

# Remove an element by its index
removed_element = my_list.pop(1)  # Remove the element at index 1
print("List after popping element at index 1:", my_list)
print("Popped element:", removed_element)


# In[3]:


#3. Write a program to demonstrate working with tuples in Python
# Creating a tuple
my_tuple = (1, 2, 3, 4, 5)

# Accessing elements of a tuple
print("Elements of the tuple:")
for element in my_tuple:
    print(element)

# Accessing elements by index
print("Element at index 2:", my_tuple[2])

# Slicing a tuple
print("Slice of the tuple:", my_tuple[1:4])

# Tuple unpacking
a, b, c, d, e = my_tuple
print("Unpacked elements:", a, b, c, d, e)

# Concatenating tuples
tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
concatenated_tuple = tuple1 + tuple2
print("Concatenated tuple:", concatenated_tuple)

# Nested tuples
nested_tuple = ((1, 2), (3, 4), (5, 6))
print("Nested tuple:", nested_tuple)

# Length of a tuple
print("Length of the tuple:", len(my_tuple))

# Checking for membership
print("Is 3 present in the tuple?", 3 in my_tuple)

# Count occurrences of an element
print("Number of occurrences of 3 in the tuple:", my_tuple.count(3))

# Finding the index of an element
print("Index of element 4 in the tuple:", my_tuple.index(4))


# In[4]:


#4. Write a program to demonstrate working with dictionaries in Python.
# Creating a dictionary
my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}

# Accessing elements of a dictionary
print("Name:", my_dict['name'])
print("Age:", my_dict['age'])
print("City:", my_dict['city'])

# Modifying elements of a dictionary
my_dict['age'] = 31
print("Modified Age:", my_dict['age'])

# Adding new key-value pairs to the dictionary
my_dict['country'] = 'USA'
print("Updated Dictionary:", my_dict)

# Removing key-value pairs from the dictionary
removed_value = my_dict.pop('city')
print("Dictionary after removing 'city':", my_dict)
print("Removed value:", removed_value)

# Checking if a key exists in the dictionary
print("'name' in dictionary?", 'name' in my_dict)
print("'city' in dictionary?", 'city' in my_dict)

# Getting the list of keys and values in the dictionary
keys_list = list(my_dict.keys())
values_list = list(my_dict.values())
print("Keys in the dictionary:", keys_list)
print("Values in the dictionary:", values_list)

# Iterating over key-value pairs in the dictionary
print("Iterating over key-value pairs:")
for key, value in my_dict.items():
    print(key, ":", value)

# Clearing the dictionary
my_dict.clear()
print("Cleared Dictionary:", my_dict)


# In[5]:


#5. Write a program to demonstrate a) arrays b) array indexing such as slicing, integer array indexing 
#and Boolean array indexing along with their basic operations in NumPy.
import numpy as np

# Creating an array
my_array = np.array([1, 2, 3, 4, 5])

# Displaying the array
print("Array:", my_array)

# Array indexing: Slicing
slice_array = my_array[1:4]
print("Slice of the array:", slice_array)

# Array indexing: Integer array indexing
index_array = my_array[[0, 2, 4]]
print("Integer array indexing:", index_array)

# Array indexing: Boolean array indexing
bool_array = my_array[my_array % 2 == 0]
print("Boolean array indexing:", bool_array)

# Basic array operations
# Addition
add_array = my_array + 5
print("Array after addition:", add_array)

# Subtraction
sub_array = my_array - 2
print("Array after subtraction:", sub_array)

# Multiplication
mul_array = my_array * 3
print("Array after multiplication:", mul_array)

# Division
div_array = my_array / 2
print("Array after division:", div_array)

# Exponentiation
exp_array = my_array ** 2
print("Array after exponentiation:", exp_array)

# Element-wise square root
sqrt_array = np.sqrt(my_array)
print("Array after square root:", sqrt_array)

# Element-wise sine
sin_array = np.sin(my_array)
print("Array after sine operation:", sin_array)


# In[6]:


#6. Write a program to compute summary statistics such as mean, median, mode, standard deviation 
#and variance of the given different types of data
import numpy as np
from scipy import stats

# Sample data
data_numeric = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data_float = [1.5, 2.7, 3.8, 4.1, 5.6, 6.2, 7.9, 8.3, 9.4, 10.0]
data_mixed = [1, 2.5, 3, 4.5, 5, 6.5, 7, 8.5, 9, 10.5]
data_categorical = ['red', 'blue', 'green', 'blue', 'red', 'red', 'green', 'green', 'red', 'blue']

# Mean
mean_numeric = np.mean(data_numeric)
mean_float = np.mean(data_float)

# Median
median_numeric = np.median(data_numeric)
median_float = np.median(data_float)

# Mode
mode_categorical = stats.mode(data_categorical)

# Standard deviation
std_numeric = np.std(data_numeric)
std_float = np.std(data_float)

# Variance
var_numeric = np.var(data_numeric)
var_float = np.var(data_float)

# Displaying summary statistics
print("Summary Statistics:")
print("Numeric Data:")
print("Mean:", mean_numeric)
print("Median:", median_numeric)
print("Standard Deviation:", std_numeric)
print("Variance:", var_numeric)
print("\nFloat Data:")
print("Mean:", mean_float)
print("Median:", median_float)
print("Standard Deviation:", std_float)
print("Variance:", var_float)
print("\nCategorical Data:")
print("Mode:", mode_categorical.mode[0])


# In[7]:


#7. Write a script named copyfile.py. This script should prompt the user for the names of two text 
#files. The contents of the first file should be the input that to be written to the second file.
def main():
    # Prompting the user for file names
    input_file = input("Enter the name of the input file: ")
    output_file = input("Enter the name of the output file: ")

    # Copying contents from input file to output file
    try:
        with open(input_file, 'r') as f_input:
            input_content = f_input.read()

        with open(output_file, 'w') as f_output:
            f_output.write(input_content)

        print("Contents of '{}' copied to '{}' successfully.".format(input_file, output_file))
    except FileNotFoundError:
        print("Error: One or both files not found.")
    except Exception as e:
        print("An error occurred:", e)


if __name__ == "__main__":
    main()


# In[8]:


#8. Write a program to demonstrate Regression analysis with residual plots on a given data set.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate some sample data
np.random.seed(0)
X = np.random.rand(100, 1) * 10
y = 3 * X.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Compute residuals
residuals = y_test - y_pred

# Plotting
plt.figure(figsize=(10, 5))

# Residuals vs. Predicted Values plot
plt.subplot(1, 2, 1)
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

# Residuals histogram
plt.subplot(1, 2, 2)
plt.hist(residuals, bins=20)
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[9]:


#9. Write a program to demonstrate the working of the decision tree-based ID3 algorithm. Use an
#appropriate data set for building the decision tree and apply this knowledge to classify a new 
#sample.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier using ID3 algorithm
clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
clf.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 6))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()

# Apply the classifier to classify a new sample
new_sample = [[5.1, 3.5, 1.4, 0.2]]  # Example new sample
predicted_class = clf.predict(new_sample)
predicted_class_name = iris.target_names[predicted_class[0]]
print("Predicted class for new sample:", predicted_class_name)


# In[ ]:


#10. Write a program to implement the Naïve Bayesian classifier for a sample training data set stored as 
#a .CSV file. Compute the accuracy of the classifier, considering few test data sets.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the dataset from CSV file
data = pd.read_csv("C:/Gskd Programs C/Python/iris.data.csv")

# Split data into features (X) and target labels (y)
X = data.drop('target', axis=1)
y = data['target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
clf = GaussianNB()

# Train the classifier
clf.fit(X_train, y_train)

# Predict the target labels for the test set
y_pred = clf.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the classifier:", accuracy)


# In[12]:


#11. Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set. Print 
#both correct and wrong predictions using Java/Python ML library classes.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the k-NN classifier
k = 3  # Number of neighbors to consider
clf = KNeighborsClassifier(n_neighbors=k)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the target labels for the test set
y_pred = clf.predict(X_test)

# Compute the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of the classifier:", accuracy)

# Print correct and wrong predictions
for i in range(len(y_test)):
    if y_test[i] == y_pred[i]:
        print("Correct prediction: Actual Class - {}, Predicted Class - {}".format(iris.target_names[y_test[i]], iris.target_names[y_pred[i]]))
    else:
        print("Wrong prediction: Actual Class - {}, Predicted Class - {}".format(iris.target_names[y_test[i]], iris.target_names[y_pred[i]]))


# In[ ]:


#12. Write a program to implement k-Means clustering algorithm to cluster the set of data stored in 
#.CSV file. Compare the results of various “k” values for the quality of clustering
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset from CSV file
data = pd.read_csv('data.csv')

# Selecting relevant features
X = data[['feature1', 'feature2']]  # Adjust features accordingly

# Define the range of k values to try
k_values = range(2, 11)

# Perform clustering for each value of k and compute silhouette score
silhouette_scores = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plotting silhouette scores for different k values
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs. Number of Clusters')
plt.xticks(k_values)
plt.grid(True)
plt.show()


# In[14]:


#13. Write a program to build Artificial Neural Network and test the same using appropriate data sets.
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential([
    Dense(10, input_shape=(4,), activation='relu'),  # Hidden layer with 10 neurons
    Dense(3, activation='softmax')  # Output layer with 3 neurons for 3 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=5, verbose=1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)


# In[ ]:




