
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

COMPANY_NAME = 'RAMOS ZINTECH LTD'

    
def get_x_y_data(df,country,type):
    """
    df: dataframe
    country: name of the country
    type: type of data to be predicted
    returns x and y data after filtering the dataframe based on the country and type
    """
    
    X = df[df['Country']==country].iloc[:,2].values.reshape(-1,1) if type == 'gdp' else df[df['Country']==country].iloc[:,1].values.reshape(-1,1)
    Y = df[df['Country']==country].iloc[:,-1].values
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # reshape Y for 2d array to be able to use imputer
    Y= Y.reshape(-1,1)
    imputer.fit(Y[:])
    Y[:]=imputer.transform(Y[:])
    # reshape Y back to 1d array
    Y=Y.reshape(-1)
    return X,Y

def encode_categorical_data():
    # encode categorical data
    # create column transformer object
    return  ColumnTransformer(transformers=
        [('one_hot_encoder', OneHotEncoder(), [0])],
        remainder='passthrough'
    )


def get_country_list(df):
    """
    df: dataframe
    """
    assert type(df) == pd.DataFrame,"Invalid dataframe"
    return list(df['Country'].unique())
    

def check_country_in_list(name,list):
    """
    name: name of the country
    list: list of countries
    returns a list of countries that match the name to be presented to the user
    """
    countries = []
    for country in list:
        # if country name matches any of the countries in the list
        if name.lower() in  country.lower():
            countries.append(country)
    return countries

def get_country(country,df):
    """
    Parameters: country, dataframe
    country: name of the country
    df: dataframe
    returns the country name if found in the dataframe
    """
    countries =  check_country_in_list(country,get_country_list(df))
    print(f"Countries found: ({len(countries)})")
    for i in range(len(countries)):
        print(f"{i+1}. {countries[i]}")
    assert len(countries) > 0,"No country found"
    input_cf = input("\nEnter number to select from \n the list of countries above: ")
    
    if(input_cf.isnumeric() and int(input_cf) <= len(countries)):
        country = countries[int(input_cf)-1]
    
    return country

def make_prediction(country,year,regressor,df,type='expense'):
    """
    country: name of the country
    year: year
    regressor: regressor object
    df: dataframe
    type: type of data to be predicted
    returns the predicted value and the x and y data together with the datasets(x,y), after filtering the dataframe based on the country and year
    and then fitting the regressor to the data
    """
    x,y = get_x_y_data(df,country,type)
    poly_feat = PolynomialFeatures(degree=2)
    x_poly = poly_feat.fit_transform(x)
    regressor.fit(x_poly,y)
    new_x =np.array(int(year), dtype=np.object ).reshape(-1,1)
    y_pred = regressor.predict(poly_feat.fit_transform(new_x))
    return new_x,y_pred,x,y,r2_score(y,regressor.predict(poly_feat.fit_transform(x)))

def plot(type,title,y_pred,new_x,x,y, labelX='Year',labelY='',score=0,res=0, year=''):
    """
    type: type of data to be predicted
    title: title of the plot
    y_pred: predicted value
    new_x: year of the prediction
    x: x data
    y: y data
    labelX: label of the x axis
    lableY: label of the y axis,
    score: r2 score
    res: result,
    year: year of the prediction
    Result: plot of the predicted value and the actual value
    """
    # convert x into 1 dimensional array horizontally
    x = x.reshape(-1)
    mymodel = np.poly1d(np.polyfit(x,y,3))
    myline = np.linspace(x[0],x[-1],100)
    plt.scatter(x,y,color='green')
    plt.scatter(new_x,y_pred,color='blue')
    plt.plot(myline,mymodel(myline),color='yellow')
    plt.title(title)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.ylim(1,max(y_pred)*2)
    # display score on the plot
    prediction_text = f"Result: {res:.4f}%" if type == 'expense' else f"Result: {res:,.2f} "
    plt.annotate(f"R2 Score: {score:.3f}",xy=(0.1,0.7),xycoords='axes fraction')
    plt.annotate(prediction_text,xy=(0.1,0.9),xycoords='axes fraction')
    plt.annotate(f"Year: {year}",xy=(0.1,0.8),xycoords='axes fraction')
    # create a legend
    plt.legend(['Actual ','Predicted','Prediction Plot'])
    plt.show()
    

def main():
    filename = ''
    try:
        # ct = encode_categorical_data() 
        
        print(f"Copyright (c) 2022 {COMPANY_NAME}")
        print("Menu: \n 1.) Predict Government Expenses \n 2.) Predict GDP Growth \n 3.) Exit")
        
        
        ch = int(input("Enter your choice from the menu above: "))
        assert ch in [1,2,3],"Invalid choice"
        c_or_reg = input("Search for country/region ").title()
        assert len(c_or_reg) > 0,"Invalid input"
        year = input("Enter a year: ")
        assert year.isnumeric() and len(year)==4,"Invalid year"
        regressor = LinearRegression()
        new_x,y_pred,x,y,score,country,type = None,[],None,None,None, '',''
        if(ch==1):
            filename = 'government_expenses_countries.csv'
            type= 'expense'
            df = pd.read_csv(filename)
            country = get_country(c_or_reg,df)
            new_x,y_pred,x,y,score = make_prediction(country,year,regressor,df)
            print(f"\nPredicted Government Expenses for {country} in {year} is {y_pred[0]:.4f} percent of GDP")
            
            
        elif(ch==2):
            filename = 'gdp.csv'
            type = 'gdp'
            df = pd.read_csv(filename)
            country = get_country(c_or_reg,df)
            new_x,y_pred,x,y,score = make_prediction(country,year,regressor,df,type)
            print(f"\nPredicted GDP Growth for {country} in {year} is {y_pred[0]:,.2f}")
    
            
        if(new_x!=None,y_pred!=[],x!=None,y!=None,score!=None):
            print(f"R-Squared Score: {score:.4f}")
            if type == 'expense':
                plot('expense',f"Predicted Government Expenses for {country} in {year}",y_pred,new_x,x,y,labelX='Year',labelY='Percent of GDP',score=score,res=y_pred[0],year=year)
            else:
                plot('gdp',f"Predicted GDP Growth for {country} in {year}",y_pred,new_x,x,y,labelX='Year',labelY='GDP Growth in dollars',score=score,res=y_pred[0],year=year)
        print("\nExiting...")
    except Exception as e:
        print(e)


if __name__ == '__main__':
    main()
   


