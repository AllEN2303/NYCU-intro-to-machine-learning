from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, roc_auc_score
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
class Preprocessor:
    def __init__(self, df):
        # Initialize the preprocessor with a DataFrame
        self.df = df

    def preprocess(self):
        # Apply various preprocessing methods on the DataFrame
        self.df = self._preprocess_numerical(self.df)
        self.df = self._preprocess_categorical(self.df)
        self.df = self._preprocess_ordinal(self.df)
        return self.df

    def _preprocess_numerical(self, df):
        # Custom logic for preprocessing numerical features goes here
        numerical_features = df.iloc[:, :17].columns.tolist()

        if numerical_features:
            scaler = StandardScaler()
            df[numerical_features] = scaler.fit_transform(df[numerical_features])
        return df

    def _preprocess_categorical(self, df):
        # Add custom logic here for categorical features
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_features:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df

    def _preprocess_ordinal(self, df):
        binary_features = df.iloc[:, 17:77].columns.tolist()
        label_encoder = LabelEncoder()
        df[binary_features] = df[binary_features].apply(lambda col: label_encoder.fit_transform(col))
        return df

# Implementing the classifiers (NaiveBayesClassifier, KNearestNeighbors, MultilayerPerceptron)

# Base classifier class
class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        # Abstract method to fit the model with features X and target y
        pass

    @abstractmethod
    def predict(self, X):
        # Abstract method to make predictions on the dataset X
        pass

# Naive Bayes Classifier

class NaiveBayesClassifier(Classifier):
    def __init__(self):
        self.phi_y = None
        self.phi_x = None
        self.y_classes = None
        self.numerical_features = None
        self.binary_features = None
        self.classes1_mean = {}
        self.classes1_std = {}
        self.classes0_mean = {}
        self.classes0_std = {}
        self.class1 = None
        self.class0 = None

    def fit(self, X, y):
        # Implement the fitting logic for Naive Bayes classifier
        self.numerical_features = X.iloc[:, :17].columns.tolist()
        self.binary_features = X.iloc[:, 17:77].columns.tolist()
        result_1_data = X[y == 1]
        for feature in self.numerical_features:
            mean_value_result_1 = result_1_data[feature].mean()
            std_value_result_1 = result_1_data[feature].std()
            self.classes1_mean[feature] = mean_value_result_1
            self.classes1_std[feature] = std_value_result_1
        for feature in self.binary_features:
            count1 = (result_1_data[feature] == 1).sum()+3
            count0 = (result_1_data[feature] == 0).sum()+3
            mean_value_result_1 = count1 / (count1 + count0)
            std_value_result_1 =  1 - mean_value_result_1
            self.classes1_mean[feature] = mean_value_result_1
        
        result_0_data = X[y == 0]
        for feature in self.numerical_features:
            mean_value_result_0 = result_0_data[feature].mean()
            std_value_result_0 = result_0_data[feature].std()
            self.classes0_mean[feature] = mean_value_result_0
            self.classes0_std[feature] = std_value_result_0
        for feature in self.binary_features:
            count1 = (result_0_data[feature] == 1).sum()+3
            count0 = (result_0_data[feature] == 0).sum()+3
            mean_value_result_0 = count1 / (count1 + count0)
            std_value_result_0 =  1 - mean_value_result_0
            self.classes0_mean[feature] = mean_value_result_0

        self.class1 = len(result_1_data) / len(X)
        self.class0 = len(result_0_data) / len(X)
        
        return self
        
    def predict(self, X):
        # Implement the prediction logic for Naive Bayes classifier
        predictions = []
        proba_array = self.predict_proba(X)
        predictions = np.apply_along_axis(lambda x: 1 if x[1] >= 0.5 else 0, axis=1, arr=proba_array)
        return predictions

    def predict_proba(self, X):
        # Implement probability estimation for Naive Bayes classifier
        class_probabilities = []
        for index, row in X.iterrows():
            prob_1 = 1.0
            prob_0 = 1.0
            for feature in self.numerical_features:
                mean_1 = self.classes1_mean[feature]
                std_1 = self.classes1_std[feature]
                pdf_1 = norm.pdf(row[feature], loc=mean_1, scale=std_1)

                mean_0 = self.classes0_mean[feature]
                std_0 = self.classes0_std[feature]
                pdf_0 = norm.pdf(row[feature], loc=mean_0, scale=std_0)
                prob_1 *= pdf_1
                prob_0 *= pdf_0

            for feature in self.binary_features:
                
                if(row[feature] == 1):
                    prob_0 *= self.classes0_mean[feature]
                    prob_1 *= self.classes1_mean[feature]
                else: 
                    prob_1 *= 1 - self.classes1_mean[feature]
                    prob_0 *= 1 - self.classes0_mean[feature]
            
            prob_1 *= self.class1
            prob_0 *= self.class0
            if(prob_1 == 0 and prob_0 == 0):
                prob_0 = 1
            total_prob = prob_1 + prob_0
            prob_1 /= total_prob
            class_probabilities.append((index, prob_1))
        proba_dataframe = pd.DataFrame(class_probabilities, columns=['Instance Index', 'Probability'])
        proba_array = proba_dataframe.values
        
        return proba_array

# K-Nearest Neighbors Classifier
class KNearestNeighbors(Classifier):
    def __init__(self, k=7):
        # Initialize KNN with k neighbors
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Store training data and labels for KNN
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # Implement the prediction logic for KNN
        proba_array = self.predict_proba(X)
        predictions = np.apply_along_axis(lambda x: 1 if x[1] >= 0.5 else 0, axis=1, arr=proba_array)
        return predictions
    
    def predict_proba(self, X):
        # Implement probability estimation for KNN
        all_distances = []

        for i_pred, X_pred in X.iterrows():
            individ_distances = []
            
            for i_fit, X_fit in self.X_train.iterrows():
                #distance = np.linalg.norm(X_pred - X_fit)
                distance = np.abs(X_pred - X_fit).sum()
                individ_distances.append((i_pred, i_fit, distance))
            all_distances.append(individ_distances)
        nn_dict = {}
        for i_pred, distances in enumerate(all_distances):
            sorted_d = sorted(distances, key=lambda x: x[2])
            nearest_neighbors = sorted_d[:self.k]
            nn_dict[i_pred] = (distances[0][0], nearest_neighbors)

        class_probabilities = []
        for key, (i_pred, neighbors) in nn_dict.items():
            instance_index = i_pred
            labels = [self.y_train[neighbor[1]] for neighbor in neighbors]
            prob_1 = labels.count(1) / len(labels)
            class_probabilities.append((instance_index, prob_1))
        proba_dataframe = pd.DataFrame(class_probabilities, columns=['Instance Index', 'Probability'])
        proba_array = proba_dataframe.values
        return proba_array


# Multilayer Perceptron Classifier
class MultilayerPerceptron(Classifier):
    def __init__(self, input_size, hidden_layers_sizes, output_size):
        # Initialize MLP with given network structure
        self.input_size = input_size
        self.hidden_layers_sizes = hidden_layers_sizes
        self.output_size = output_size
        self.learning_rate = 0.8
        self.proba = []
        self.weights = []
        self.biases = []
        layer_sizes = [self.input_size] + self.hidden_layers_sizes + [self.output_size]

        for i in range(1, len(layer_sizes)):
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i])))


    def fit(self, X, y, epochs = 300 , learning_rate = 0.8):
        # Implement training logic for MLP including forward and backward propagation
        targets = np.zeros((len(y), self.output_size))
        targets[np.arange(len(y)), y] = 1
        for epoch in range(epochs):
            total_error = 0
            for i in range(len(X)):
                X_batch, target_batch = X[i:i+1], targets[i:i+1]
                activations = self._forward_propagation(X_batch)
                total_error += self._calculate_error(activations[-1], target_batch)
                self._backward_propagation(activations, target_batch)
    def _calculate_error(self, output, target):
        error = np.sum(0.5 * np.square(target - output)) / len(target)
        return error
    def predict(self, X):
        # Implement prediction logic for MLP
        proba_array = self.predict_proba(X)
        predictions = np.argmax(proba_array, axis=1)
        return predictions

    def predict_proba(self, X):
        # Implement probability estimation for MLP
        activations = self._forward_propagation(X)
        return activations[-1]
        
    def _forward_propagation(self, X):
        # Implement forward propagation for MLP
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = 1 / (1 + np.exp(-z))
            activations.append(a)
        output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        exp_values = np.exp(output - np.max(output, axis=1, keepdims=True))
        final_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        activations.append(final_output)
        return activations

    def _backward_propagation(self, output, target):
        # Implement backward propagation for MLP
        delta_output = (1-output[-1])* (output[-1]) * (target - output[-1])

        for i in range(len(self.weights) - 1, 0, -1):
            delta_hidden = (1-output[i]) * (output[i]) * np.dot(delta_output, self.weights[i].T)

            self.weights[i] += self.learning_rate * np.dot(output[i].T, delta_output)
            self.biases[i] += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)

            delta_output = delta_hidden

        self.weights[0] += self.learning_rate * np.dot(output[0].T, delta_output)
        self.biases[0] += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)

# Function to evaluate the performance of the model
def evaluate_model(model, X_test, y_test):
    # Predict using the model and calculate various performance metrics
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    mcc = matthews_corrcoef(y_test, predictions)

    # Check if the model supports predict_proba method for AUC calculation
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_test)
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, proba[:, 1])
        else:  # Multiclass classification
            auc = roc_auc_score(y_test, proba, multi_class='ovo')
    else:
        auc = None

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'mcc': mcc,
        'auc': auc
    }
# Main function to execute the pipeline
def main():
    # Load trainWithLable data
    df = pd.read_csv('trainWithLabel.csv')
    
    numerical_features = df.iloc[:, :17].columns.tolist()
    binary_features = df.iloc[:, 17:77].columns.tolist()

    # Preprocess the training data
    # preprocessor = Preprocessor(df)
    # df = preprocessor.preprocess()
    # Define the models for classification
    models = {'Naive Bayes': NaiveBayesClassifier()#,
              ,'KNN': KNearestNeighbors(),
              'MLP': MultilayerPerceptron(77,[64, 32, 24, 8],2)
    }
    X_train = df.drop('Outcome', axis=1)
    y_train = df['Outcome']
    # Split the dataset into features and target variable
    # Perform K-Fold cross-validation
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = []
    for name, model in models.items():
        for fold_idx, (train_index, val_index) in enumerate(kf.split(X_train), start=1):
            #if(fold_idx != 1): continue
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index] #modify iloc
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
            X_train_fold = X_train_fold.copy()
            X_val_fold = X_val_fold.copy()
            for feature in numerical_features:
                num = X_train_fold[feature].median()
                for i in train_index:
                    if pd.isnull(X_train_fold[feature][i]):
                        if i == 0: X_train_fold.loc[i, feature] = num
                        else: X_train_fold.loc[i, feature] = num
                    num = X_train_fold.loc[i, feature]

            for feature in binary_features:
                probability = X_train_fold[feature].sum() / X_train_fold[feature].count()
                for i in train_index:
                    if pd.isnull(X_train_fold[feature][i]):
                        fill_value = np.random.choice([0, 1], p=[1 - probability, probability])
                        X_train_fold.loc[i, feature] = fill_value
            
            for feature in numerical_features:
                num = X_val_fold[feature].median()
                for i in val_index:
                    if pd.isnull(X_val_fold[feature][i]):
                        if i == 0: X_val_fold.loc[i, feature] = num
                        else: X_val_fold.loc[i, feature] = num
                    num = X_val_fold.loc[i, feature]

            for feature in binary_features:
                probability = X_val_fold[feature].sum() / X_val_fold[feature].count()
                for i in val_index:
                    if pd.isnull(X_val_fold[feature][i]):
                        fill_value = np.random.choice([0, 1], p=[1 - probability, probability])
                        X_val_fold.loc[i, feature] = fill_value
            preprocessor = Preprocessor(X_train_fold)
            X_train_fold = preprocessor.preprocess()
            preprocessor = Preprocessor(X_val_fold)
            X_val_fold = preprocessor.preprocess()    
            model.fit(X_train_fold, y_train_fold)
            fold_result = evaluate_model(model, X_val_fold, y_val_fold)

            fold_result['model'] = name
            fold_result['fold'] = fold_idx
            cv_results.append(fold_result)

    # Convert CV results to a DataFrame and calculate averages
    cv_results_df = pd.DataFrame(cv_results)
    avg_results = cv_results_df.groupby('model').mean().reset_index()
    avg_results['model'] += ' Average'
    all_results_df = pd.concat([cv_results_df, avg_results], ignore_index=True)

    # Adjust column order and display results
    all_results_df = all_results_df[['model', 'accuracy', 'f1', 'precision', 'recall', 'mcc', 'auc']]

    print("Cross-validation results:")
    print(all_results_df)

    # Save results to an Excel file
    all_results_df.to_excel('cv_results.xlsx', index=False)
    print("Cross-validation results with averages saved to cv_results.xlsx")
    # Load the test dataset, assuming you have a test set CSV file without labels

    df_ = pd.read_csv('testWithoutLabel.csv')

    for feature in numerical_features:
        num = df_[feature].median()
        for i in range(len(df_)):
            if pd.isnull(df_[feature][i]):
                if i == 0: df_.loc[i, feature] = num
                else: df_.loc[i, feature] = num
            num = df_.loc[i, feature]

    for feature in binary_features:
        probability = df_[feature].sum() / df_[feature].count()
        for i in range(len(df_)):
            if pd.isnull(df_[feature][i]):
                fill_value = np.random.choice([0, 1], p=[1 - probability, probability])
                df_.loc[i, feature] = fill_value

    preprocessor_ = Preprocessor(df_)
    X_test = preprocessor_.preprocess()

    # Initialize an empty list to store the predictions of each model
    predictions = []

    # Make predictions with each model
    for name, model in models.items():
        model_predictions = model.predict(X_test)
        predictions.append({
            'model': name,
            'predictions': model_predictions
        })

    # Convert the list of predictions into a DataFrame
    predictions_df = pd.DataFrame(predictions)

    # Print the predictions
    print("Model predictions:")
    print(predictions_df)

    # Save the predictions to an Excel file
    predictions_df.to_csv('test_results.csv', index=False)
    print("Model predictions saved to test_results.xlsx")

if __name__ == "__main__":
    main()
