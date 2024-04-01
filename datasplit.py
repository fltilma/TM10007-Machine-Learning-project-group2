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

#pick 8 random samples from both sets. using random state for reproducability
malignant_valdation = malignant.sample(n=8, random_state=1)
benign_validation = benign.sample(n=8, random_state=1)

#remove the validation samples from the train set
malignant_train = malignant.drop(malignant_valdation.index).reset_index(drop=True)
bening_train = benign.drop(benign_validation.index).reset_index(drop=True)

#combining the dfs into workable sets and writing them to csv
validation = pd.concat([malignant_valdation,benign_validation])
train = pd.concat([malignant_train,bening_train])
validation.to_csv('validation_data.csv', index=False)
train.to_csv('train_data.csv', index=False)
