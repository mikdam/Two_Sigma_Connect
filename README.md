# Two Sigma Connect
This is a machine learning solution I developed for a Kaggle competetion [Two Sigma Connect: Rental Listing Inquiries](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries)

The dataset can be downloaded from [here](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries/data).

Check out [my Kaggle profile](https://www.kaggle.com/mikdam/competitions) for this submission score and ranking.

### Feature Engineering:

* Clean the text of display address feature by removing noise and normalising the addresses list, then encode it using emperical Bayesian statistical method to code the feature as probability of the target variable.

* Extract features from the text of Description feature. First the text is been tokenised, lammatise and vectorised using TFIDF Vectoriser. Then Latent Dirichlet Allocation topic modelling to discover the topics distribution and use this distribution as features.

* Cleaning the Features feature and encoding it as probability of the target variable.

* Use the Latitude and Longitude features to segement the covered area into small squares which will be numbered and treated as categorical variable. This variable will be encoded using emperical Bayesian statistical method as probability of the target variable. It also has been combine with other features.

* Combine a number of the categorical  features to create new features which will be encoded, along side some other features, using emperical Bayesian as probability of the target variable.

* Extract a number of various numerical features.

### Handling Imblanced Dataset

As the dataset is imbalanced, a number of techniques for over and under-sample have been used: Random Undersampling,  Adaptive Synthetic Sampling Approach (ADASYN) oversampling and Tome Links undersampling. imblearn package has been used.

### Stacking models
A stacked model of three layers has been built. The  first layer uses Random Forest to combine the prdiction probabilities of a number of models. The classification problem has been converted to binary classification problems using One-vs-the-Rest (XGB_Bin_OvR) and One-vs-One (XGB_Bin_OvO) strategies. One model used the problem in its multiclass form (XGB_Multi). Various combination of oversampling and undersampling techniques with the three stacked models above.
