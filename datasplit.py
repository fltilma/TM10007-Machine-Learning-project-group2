'''Here, we create a data split. We will save a validation set for testing, when we are satisfied with the algorithm.
Since we will use LOOCV, we do not need a separate training set.'''
import pandas as pd
#data = load_data()
data = pd.read_csv(r"C:\Users\delan\Downloads\TM10007-Machine-Learning-project-group2\worcliver\Liver_radiomicFeatures.csv")

malignant = data[data['label'] == 'malignant']
benign = data[data['label'] == 'benign']

mal = len(malignant.index)
ben = len(benign.index)
'''We see that the amount of bening and malignant is relatively the same. Therefore,
 our validation set wil contain 8 bening samples and 8 malignant samples'''

#pick 8 random samples from both sets for validation
malignant_valdation = malignant.sample(n=8, random_state=1)
benign_validation = benign.sample(n=8, random_state=1)

#remove the validation samples from the test and train set
malignant_testtrain = malignant.drop(malignant_valdation.index).reset_index(drop=True)
benign_testtrain = benign.drop(benign_validation.index).reset_index(drop=True)

#pick 8 random samples from both sets for testing
malignant_test = malignant_testtrain.sample(n=8, random_state=1)
benign_test = benign_testtrain.sample(n=8, random_state=1)

#remove the train samples from the train set
malignant_train = malignant_testtrain.drop(malignant_test.index).reset_index(drop=True)
benign_train = benign_testtrain.drop(benign_test.index).reset_index(drop=True)

#combining the dfs into workable sets and writing them to csv
validation = pd.concat([malignant_valdation,benign_validation])
test = pd.concat([malignant_test,benign_test])
train = pd.concat([malignant_train,benign_train])
validation = validation.loc[:, validation.apply(pd.Series.nunique) != 1]
test = test.loc[:, test.apply(pd.Series.nunique) != 1]
train = train.loc[:, train.apply(pd.Series.nunique) != 1]
validation.to_csv('validation_data.csv', index=False)
test.to_csv('test_data.csv', index=False)
train.to_csv('train_data.csv', index=False)
