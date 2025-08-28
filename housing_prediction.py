'''
I built this project to learn about Regression Models.
In this project I have Explored Linear Regression, Polynomial Regression, Lasso Regression and Ridge Regression.
I have used Lasso and Ridge Regression for reducing Overfitting and they are called Regularisation Methods.
'''
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
import sys


class House_prediction():
    def __init__(self,location):
        try:
            self.df=pd.read_csv(location, header=None,delim_whitespace='True')
            self.X=self.df.iloc[:,:-1]
            self.y=self.df.iloc[:,-1]
            self.X_train,self.X_test,self.y_train,self.y_test=train_test_split(self.X,self.y,test_size=0.2,random_state=42)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')
    def linear_regression(self):
        try:
            self.regressor=LinearRegression()
            self.regressor.fit(self.X_train, self.y_train)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')
    def polynomial_regression(self):
        try:
            self.poly=PolynomialFeatures(degree=2)
            self.poly_X_train = self.poly.fit_transform(self.X_train)
            self.poly_X_test = self.poly.fit_transform(self.X_test)
            self.poly_regressor = LinearRegression()
            self.poly_regressor.fit(self.poly_X_train, self.y_train)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')

    def lasso_regression(self):
        try:
            self.lasso=Lasso(alpha=0.01,random_state=42)
            self.lasso.fit(self.poly_X_train, self.y_train)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')
    def ridge_regression(self):
        try:
            self.ridge=Ridge(alpha=0.01,random_state=42)
            self.ridge.fit(self.poly_X_train, self.y_train)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')
    def model_performance(self):
        try:
            models=pd.DataFrame()
            models['Model Training Performance']=[self.regressor.score(self.X_train,self.y_train),
                                                  self.poly_regressor.score(self.poly_X_train,self.y_train),
                                                  self.lasso.score(self.poly_X_train, self.y_train),
                                                  self.ridge.score(self.poly_X_train, self.y_train)]

            models['Model Testing Performance']=[self.regressor.score(self.X_test,self.y_test),
                                                 self.poly_regressor.score(self.poly_X_test,self.y_test),
                                                 self.lasso.score(self.poly_X_test, self.y_test),
                                                 self.ridge.score(self.poly_X_test, self.y_test)]
            models['Model Mean square Error']=[mean_squared_error(self.y_test,self.regressor.predict(self.X_test)),
                                               mean_squared_error(self.y_test,self.poly_regressor.predict(self.poly_X_test)),
                                               mean_squared_error(self.y_test,self.lasso.predict(self.poly_X_test)),
                                               mean_squared_error(self.y_test,self.ridge.predict(self.poly_X_test))]
            models.index=['Linear Regression','Polynomial Regression','Lasso Regression','Ridge Regression']
            print(models)
        except Exception as e:
            error_type, error_msg, err_line = sys.exc_info()
            print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')
if __name__=='__main__':
    try:
        location='C:\\Users\\mdhan\\OneDrive\\Desktop\\Projects\\ML\\Regression\\Linear Regression\\housing.csv'
        obj=House_prediction(location)
        obj.linear_regression()
        obj.polynomial_regression()
        obj.lasso_regression()
        obj.ridge_regression()
        obj.model_performance()
    except Exception as e:
        error_type,error_msg,err_line=sys.exc_info()
        print(f'Error from line -> {err_line.tblineno}  -> type -> {error_type} -> error message {error_msg}')

