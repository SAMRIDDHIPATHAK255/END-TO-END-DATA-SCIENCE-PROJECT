# END-TO-END-DATA-SCIENCE-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: SAMRIDDHI PATHAK

*INTERN ID*: CT08DL700

*DOMAIN*: DATA SCIENCE

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

This end-to-end data science project focuses on building and deploying a machine learning model to predict passenger survival on the Titanic. The project begins with data collection, where we use a cleaned Titanic dataset containing information such as age, fare, sex, and embarkation port of passengers. This data is read into a pandas DataFrame for exploration and analysis. The goal is to predict whether a passenger survived the disaster, using these features. The dataset is split into features (independent variables) and the target variable, "survived," which indicates the survival status of each passenger. The features include both numerical and categorical variables, requiring preprocessing to prepare the data for machine learning.

To preprocess the data, we build a pipeline that uses a `ColumnTransformer`. Numerical features such as age and fare are standardized using `StandardScaler`, while categorical features like sex and embarkation point are converted into numerical format using one-hot encoding. This preprocessing step ensures that the model can interpret all input features correctly and uniformly. A `RandomForestClassifier` is used as the predictive model due to its ability to handle both numerical and categorical features efficiently, its robustness against overfitting, and its high performance on classification tasks. The model is trained using a train-test split approach, where 80% of the data is used for training and 20% is used to evaluate the model's performance. After training, the model achieves a good accuracy score on the test set, demonstrating its ability to generalize well on unseen data.

Once the model is trained and evaluated, it is saved to disk using the `joblib` library, allowing it to be reused without retraining. The next step is deployment, which involves building a Flask web application that can serve the model predictions. The Flask app loads the saved model and provides a user interface through an HTML form. Users can input passenger details such as age, fare, sex, and port of embarkation into the form. When the form is submitted, the app processes the input, creates a DataFrame compatible with the model's expected input, and passes it to the model for prediction. The prediction result is displayed back on the same web page, showing whether the model predicts the passenger would have survived or not.

This web application transforms the data science model into a usable service, allowing real-time interaction through a browser. The project exemplifies a complete machine learning lifecycle: from data preparation and model development to integration and deployment. It demonstrates the importance of not only building a strong model but also delivering it in a way that can be accessed and understood by end users. By using Flask, a lightweight web framework, the deployment is kept simple and accessible, making it ideal for small projects or educational purposes. The result is a functional, interactive, and informative application that showcases how data science can provide insights and predictions in a user-friendly format.
