import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import seaborn as sns

df = pd.read_csv("movie_metadata.csv")

# Select only important attributes for the dataframe for predict the gross of the movie
df = df[["num_critic_for_reviews", "duration", "director_facebook_likes", "actor_3_facebook_likes","actor_1_facebook_likes" ,"num_voted_users","cast_total_facebook_likes",
         "facenumber_in_poster", "num_user_for_reviews", "language", "country", "budget", "actor_2_facebook_likes", "imdb_score","aspect_ratio", "movie_facebook_likes", "gross"]]
df.head(5)

print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())
print('\nUnique values : \n',df.nunique())
print('Number of null values of each column')
df.isna().sum()

# Drop all the null values of country and language columns
df = df.dropna(subset=['country','language'],inplace=False)

# Encode the country values to a numberical value
country = LabelEncoder() #define the encoder
df['country'] = country.fit_transform(df['country'])
df["country"].unique()

# Encode the language values to a numerical value
language = LabelEncoder()  #define the enocoder
df['language'] = language.fit_transform(df['language'])
df["language"].unique()

### Find out most Important Features ####
# Plotting the Correlation between the numerical values of the Dataset
correlations = df.corr()
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(correlations, annot=True, cmap="YlGnBu", linewidths=.5)

# According to the heatmap result and nature of the predicted variable , drop other variables from the dataframe while keeping other important variables as independent variables
df = df.drop(['actor_3_facebook_likes','cast_total_facebook_likes','facenumber_in_poster','language','aspect_ratio'], axis = 1)

### Pre Processing

## Method for remove the reocrds of countries which are under a given limit
def cutoff_countries(countries, limit):
   country_map = {}
   for i in range(len(countries)):
       if countries.values[i] < limit:
           country_map[countries.index[i]] = 'Other'
       else:
           country_map[countries.index[i]] = countries.index[i]
   return country_map

# Remove countries which are having records of 80       
country_map = cutoff_countries(df.country.value_counts(), 80)
df['country'] = df['country'].map(country_map)
df.country.value_counts()
df = df[df['country'] != 'Other']

# Filling missing values
df['gross'] = df['gross'].fillna(df['gross'].mean(), inplace=False)
df['num_user_for_reviews'] = df['num_user_for_reviews'].fillna(df['num_user_for_reviews'].mean(), inplace=False)
df['imdb_score'] = df['imdb_score'].fillna(df['imdb_score'].mean(), inplace=False)
df['budget'] = df['budget'].fillna(df['budget'].mean(), inplace=False)
df['num_voted_users'] = df['num_voted_users'].fillna(df['num_voted_users'].mean(), inplace=False)
df['num_critic_for_reviews'] = df['num_critic_for_reviews'].fillna(df['num_critic_for_reviews'].mean(), inplace=False)
df['movie_facebook_likes'] = df['movie_facebook_likes'].fillna(df['movie_facebook_likes'].mean(), inplace=False)
df['actor_1_facebook_likes'] = df['actor_1_facebook_likes'].fillna(df['actor_1_facebook_likes'].mean(), inplace=False)
df['actor_2_facebook_likes'] = df['actor_2_facebook_likes'].fillna(df['actor_2_facebook_likes'].mean(), inplace=False)
df['duration'] = df['duration'].fillna(df['duration'].mean(), inplace=False)
df['director_facebook_likes'] = df['director_facebook_likes'].fillna(df['director_facebook_likes'].mean(), inplace=False)

df.head()