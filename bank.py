from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def main():
    bank_data = pd.read_csv('bank-full.csv',';')
    train = 513 # Best so far is 513, with 0.8844914761286858

    bank_X = bank_data.iloc[:,:16]
    bank_Y = bank_data.iloc[:,16:]

    bank_X_cont = bank_X[['age','balance','day','duration','campaign','pdays','previous']].values
    bank_X_cat = bank_X[['job','marital','education','default','housing','loan','contact','month','poutcome']].values

    # For now we're using OneHotEncoding, but eventually I want to try label encoding
    enc = OneHotEncoder()
    enc.fit(bank_X_cat)
    bank_X_cat = enc.transform(bank_X_cat).toarray()
    enc.fit(bank_Y)
    bank_Y = enc.transform(bank_Y).toarray()

    bank_data = []
    for index in range(0,len(bank_X_cat)):
        a = np.concatenate((bank_X_cat[index].astype(int),bank_X_cont[index].astype(int)))
        bank_data.append(a)

    bank_data = np.asarray(bank_data)

    bank_data_train = bank_data[0:train]
    bank_data_test = bank_data[train:]

    bank_Y_train = bank_Y[0:train]
    bank_Y_test = bank_Y[train:]

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(bank_data_train,bank_Y_train)

    print(neigh.score(bank_data_train,bank_Y_train))
    print(neigh.score(bank_data_test,bank_Y_test))


if __name__ == '__main__':
    main()
