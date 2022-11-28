import pickle
import numpy as np

# Get the user inputs
budget = float(input("Enter Budget: "))
country = input("Enter Country: ")
imdb_score = float(input("Enter Imdb Score: "))
num_user_for_reviews = int(input("Enter Number of reviewed users: "))


# Assigning the meadian values for other independent variables
num_critic_for_reviews = 115.0
duration = 103.0
director_facebook_likes = 53.0
actor_1_facebook_likes = 1000.0
num_voted_users = 36989.5
actor_2_facebook_likes = 628.5
movie_facebook_likes = 168.5

# Load the expoerted model


def load_model():
    with open('randomforest_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

random_forest_reg = data["model"]
country_le = data["country_le"]

# Define the array with all input parameters
X = np.array([[num_critic_for_reviews, duration, director_facebook_likes, actor_1_facebook_likes,
               num_voted_users, num_user_for_reviews, country, budget, actor_2_facebook_likes, imdb_score, movie_facebook_likes]])
X[:, 6] = country_le.transform(X[:, 6])
X = X.astype(float)

# Make the prediction
gross = random_forest_reg.predict(X)
result = "{:.2f}".format(gross[0])
print("Predicted Gross : $", result)
