from ml_analyze_country import *
import pytest as pt
filename = 'government_expenses_countries.csv'
filename2 = 'gdp.csv'
def check_if_y_is_1_dimensional(y):
    assert y.ndim == 1, 'get_x_y_data() should return a 1-dimensional array for Y'



def check_get_x_y_data(df):
    # check if x and y are numpy arrays
    assert type(get_x_y_data(df,'Philippines',type="gdp")[0]) == np.ndarray, 'get_x_y_data() should return a numpy array for X'
    assert type(get_x_y_data(df,'Philippines',type="gdp")[1]) == np.ndarray, 'get_x_y_data() should return a numpy array for Y'
    check_y_for_NaN(get_x_y_data(df,'Thailand',type="gdp")[1])

def check_y_for_NaN(y):
    assert np.isnan(y).any() == False, 'get_x_y_data() should not return any NaN values in Y'


def test_get_x_y_data():
    global filename
    check_get_x_y_data(pd.read_csv(filename))
    check_if_y_is_1_dimensional(get_x_y_data(pd.read_csv(filename),'Philippines',type="expense")[1])


def check_x_and_ct(ct):
    assert type(ct) == ColumnTransformer, 'encode_categorical_data() should return a ColumnTransformer object'


def test_get_country_list():
    global filename
    assert type(get_country_list(pd.read_csv(filename))) == list, 'get_country_list() should return a list of countries'
    assert type(get_country_list(pd.read_csv(filename2))) == list, 'get_country_list() should return a list of countries'
def test_encode_categorical_data():
    global filename
    ct = encode_categorical_data()
    check_x_and_ct(ct)
    

def test_check_country_in_list():
    global filename
    assert type(check_country_in_list('philippines',get_country_list(pd.read_csv(filename)))) == list, 'check_country_in_list() should return a list of countries'
    assert type(check_country_in_list('philippines',get_country_list(pd.read_csv(filename2)))) == list, 'check_country_in_list() should return a list of countries'
    assert type(check_country_in_list('thailand',get_country_list(pd.read_csv(filename)))) == list, 'check_country_in_list() should return a list of countries'

pt.main(["-x","-v","-vv", "test_analyze_country.py"])