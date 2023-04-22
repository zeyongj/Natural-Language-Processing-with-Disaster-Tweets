# Natural-Language-Processing-with-Disaster-Tweets

This [Kaggle project](url: https://www.kaggle.com/competitions/nlp-getting-started/overview) aims to build a machine learning model to predict which tweets are about real disasters and which ones are not. The dataset consists of 10,000 tweets that were hand classified. The challenge is to create a model that can distinguish between real disaster tweets and those that are not, despite the use of metaphorical language or potentially offensive content.

## Competition Description

Twitter has become an important communication channel during emergencies. The ability to announce an emergency in real-time makes it an attractive platform for disaster relief organizations and news agencies to monitor. However, determining whether a tweet is actually announcing a disaster can be challenging, especially for machines.

The goal of this project is to build a machine learning model that can accurately predict if a given tweet is about a real disaster (1) or not (0).

## Dataset

The dataset for this competition contains potentially profane, vulgar, or offensive text. The necessary files include:

- `train.csv`: The training set
- `test.csv`: The test set
- `sample_submission.csv`: A sample submission file in the correct format

Each sample in the train and test set contains the following information:
- The text of a tweet
- A keyword from that tweet (may be blank)
- The location the tweet was sent from (may be blank)

### Columns

- `id`: A unique identifier for each tweet
- `text`: The text of the tweet
- `location`: The location the tweet was sent from (may be blank)
- `keyword`: A particular keyword from the tweet (may be blank)
- `target`: In `train.csv` only, this denotes whether a tweet is about a real disaster (1) or not (0)

## Model

This project uses a Logistic Regression model with TfidfVectorizer for feature extraction. The model achieved an F1 score of 0.78516. The following steps were taken to preprocess the data and train the model:

1. Remove URLs from the text
2. Convert the text to lowercase
3. Remove non-alphanumeric characters
4. Combine the keyword and text columns
5. Split the data into a training and validation set
6. Convert the text to a tf-idf matrix
7. Train a Logistic Regression model
8. Evaluate the model using accuracy and F1 score

## Results and Conclusion

The Logistic Regression model, combined with the TfidfVectorizer, achieved an F1 score of 0.78516. This performance indicates that the model is reasonably effective at predicting whether a tweet is about a real disaster or not. Further improvements could potentially be made by exploring more advanced natural language processing techniques or using more complex machine learning models.

## How to run the code

1. Install the required libraries:
   - pandas
   - numpy
   - re
   - scikit-learn
2. Load the datasets (`train.csv` and `test.csv`) in the same directory as the code.
3. Run the provided code to preprocess the data, train the model, and make predictions on the test set.
4. The predictions will be saved in a `submission.csv` file in the correct format.

## License

This project is licensed under the MIT License. The MIT License is a permissive open source license that allows for free use, copying, modification, and distribution of the software, as long as the copyright notice and permission notice are included in all copies or substantial portions of the software. This license is suitable for both academic and commercial projects.

## Reference

Howard, A., Devrishi, Phil Culliton, & Guo, Y. (2019). Natural Language Processing with Disaster Tweets. Kaggle. Retrieved from https://kaggle.com/competitions/nlp-getting-started .

## Author

Zeyong Jin

April 21st, 2023

