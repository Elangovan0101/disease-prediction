# Disease Prediction System 🩺🔍
![Disease Prediction](https://www.pantechelearning.com/wp-content/uploads/2021/12/GettyImages-928162118.jpg)

## Disease Prediction System

### Objective 🎯
The primary objective of this project is to create an efficient disease prediction system based on symptoms input by the user. The system uses multiple machine learning models to predict the most probable disease.

### Project Details 📋

#### Key Features ✨
- **User Input**: Users can enter symptoms in a comma-separated format.
- **Model Training**: Three different models (SVM, Naive Bayes, and Random Forest) are trained on the dataset.
- **Cross-validation**: The models are evaluated using cross-validation to ensure accuracy.
- **Majority Voting**: The final prediction is made using a majority voting system that combines the predictions of all three models.

### Project Summary 📝

#### Project Description ℹ️
The Disease Prediction System is a Flask application designed to predict diseases based on user-provided symptoms. It utilizes three different machine learning algorithms to make predictions and combines their outputs to provide the final prediction.

#### Objective 🌟
The main goal is to automate disease prediction by taking into account the symptoms provided by the user and leveraging the strengths of multiple machine learning models.

#### Key Project Details 🛠️
- **SVM Model**: Support Vector Machine is used for classification.
- **Naive Bayes Model**: Gaussian Naive Bayes is used for its simplicity and effectiveness with small datasets.
- **Random Forest Model**: Random Forest Classifier is used for its high accuracy and ability to handle large datasets.
- **Cross-validation**: 10-fold cross-validation is used to evaluate the models.
- **Confusion Matrix**: Confusion matrices are generated for each model to visualize performance.

### Flask Application 🖥️

#### Overview
The Flask application serves as the user interface for the Disease Prediction System. It handles user inputs, processes the data using the trained machine learning models, and displays the results in a user-friendly format.

#### Routes
- **Home Route (`/`)**: Displays the input form where users can enter symptoms.
- **Prediction Route (`/predict`)**: Processes the input symptoms, predicts the disease using the trained models, and displays the results.

#### Templates
- **index.html**: The main page where users input their symptoms.
- **result.html**: The results page that displays the predictions from the models.

#### Static Files
- **styles.css**: Contains the CSS styles for the application to ensure a visually appealing interface.

### Results 📊

#### Model Evaluation 📈
The application evaluates each model using cross-validation and prints out the mean accuracy for each. Confusion matrices are used to display the performance of each model on the test data.

#### Example Output 📅
An example output will show the accuracy of each model on both train and test data, along with the confusion matrix for each model. The final prediction is displayed based on the majority vote of the three models.

### Conclusion 🚀
The Disease Prediction System effectively automates the prediction of diseases based on symptoms, leveraging multiple machine learning models to ensure high accuracy. This helps in early detection and timely treatment of diseases.

### Project Execution 📑

1. **Load Data**: Load the training data and preprocess it.
2. **Train Models**: Train SVM, Naive Bayes, and Random Forest models on the data.
3. **Evaluate Models**: Evaluate the models using cross-validation and confusion matrices.
4. **User Input**: Collect user input for symptoms through the Flask application.
5. **Predict Disease**: Predict the disease based on user input using the trained models.
6. **Display Results**: Display the predictions in an easy-to-understand format through the Flask application.

### Challenges and Future Work 🛠️

- **Scalability**: Enhance the system to handle larger datasets and more complex symptoms.
- **User Interface**: Improve the UI for better user experience and add more customization options.

### Practical Application 🌐
This system can be used by healthcare providers to assist in the early detection of diseases based on symptoms, providing a valuable tool for diagnosis and treatment planning.

### How to Run the Project 🚀

1. Clone the repository: `(https://github.com/yourusername/disease-prediction-system.git)`
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Flask application with `python app.py`.
4. Open the generated URL in your web browser to use the Disease Prediction System.

### License 📜
This project is licensed under the MIT License.
