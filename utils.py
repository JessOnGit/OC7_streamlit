"""
Utils for streamlit app
"""

import streamlit as st
import pandas as pd
import joblib
import requests
import numpy as np


# >> Prepare cached environment
# Load Data
@st.cache_data
def load_dataset():
    # Import X_test et y_test
    X_test = pd.read_csv('data_test_clean/X_test_sample.csv')
    y_test = pd.read_csv('data_test_clean/y_test_sample.csv')
    data = pd.concat((X_test, y_test), axis=1)
    return X_test, y_test, data


# Get features general values
@st.cache_data
def get_general_features_values():
    return pd.read_csv('refs/min_max_mean_med.csv')


@st.cache_data
def get_median_scores():
    sc0 = pd.read_csv('refs/scores0.csv')
    sc1 = pd.read_csv('refs/scores1.csv')
    median_sc0 = sc0.median()[0]
    median_sc1 = sc1.median()[0]
    return sc0, sc1, median_sc0, median_sc1


@st.cache_data
def get_shap_explainer():
    save_directory = 'refs/shap_explainer.sav'
    loaded_explainer = joblib.load(save_directory)
    return loaded_explainer


@st.cache_data
def get_shap_values(_explainer, X_test):
    return _explainer(X_test.drop('SK_ID_CURR', axis=1))[:, :, 1]


# Get Clients info
def clients_real(id_client, data):
    df_client = data.loc[data.SK_ID_CURR == id_client].drop(['SK_ID_CURR', 'true'], axis=1)
    true = data.loc[data.SK_ID_CURR == id_client, 'true'].values[0]
    return df_client, true


def get_prediction(df_client_tolist, url_api):
    response = requests.post(url_api, json={'data': df_client_tolist})
    if response.status_code == 200:
        result = response.json()
        pred = result['pred']
        proba = result['proba']
        return pred, proba
    else:
        st.error('Error at api call')
        return None


def shap_explanation(df_client, explainer):
    sv = explainer(df_client)
    base_value = sv[:, :, 1][0].base_values
    client_score = np.round(sv[:, :, 1][0].values.sum() + base_value, 2)
    client_data = pd.DataFrame({'feature': df_client.columns.tolist(),
                                'feature_score': np.round(sv[:, :, 1][0].values, 2),
                                'sv_abs': np.abs(sv[:, :, 1][0].values),
                                'feature_value': np.round(sv[:, :, 1][0].data, 2)})
    client_data.sort_values(by='sv_abs', ascending=False, inplace=True)
    client_data.drop('sv_abs', axis=1, inplace=True)
    return client_score, client_data


def get_clients_info(df_client, url_api, explainer):
    # Call api predictions for selected client
    pred, proba = get_prediction(df_client.values.tolist(), url_api)
    # Shap explanation
    client_score, client_data = shap_explanation(df_client, explainer)
    return pred, proba, client_score, client_data


# Create 3 sliders with correct dtypes for changing variables
def create_sliders_variables(client_data, mini_maxi_features):
    features_pos = client_data.loc[client_data['feature_score'] > 0].reset_index().drop('index', axis=1)
    sliders = {}
    for i in range(3):
        # Get min and max from general feature info
        val = features_pos.loc[i, 'feature_value']
        name = features_pos.loc[i, 'feature']
        slider_dict = {'label': name}
        fmin = mini_maxi_features.loc[mini_maxi_features.features == name, 'min'].values[0]
        fmax = mini_maxi_features.loc[mini_maxi_features.features == name, 'max'].values[0]

        # Build dict with correct dtype
        if (int(val) == val) and (int(fmin) == fmin) and (int(fmax) == fmax):
            slider_dict['val'] = int(val)
            slider_dict['min'] = int(fmin)
            slider_dict['max'] = int(fmax)
            slider_dict['step'] = 1
        else:
            slider_dict['val'] = float(val)
            slider_dict['min'] = float(fmin)
            slider_dict['max'] = float(fmax)
            slider_dict['step'] = 0.1
        sliders['slider' + str(i + 1)] = slider_dict
    return sliders


def calc_risk_level(client_score, median_sc0, median_sc1, c1, c2):
    risk_level = 'Medium'
    risk_color = 'grey'
    if client_score < median_sc0:
        risk_level = 'Low'
        risk_color = c1
    elif client_score > median_sc1:
        risk_level = 'High'
        risk_color = c2
    return risk_level, risk_color


def update_page_status(status):
    st.session_state.status = status


def create_slider(sd, slider_key):
    return st.sidebar.slider(label=sd['label'], min_value=sd['min'], max_value=sd['max'],
                             value=sd['val'], step=sd['step'], key=slider_key)


def highlight_scores_values(val, c2):
    if val > 0:
        return f'background-color: {c2}'
    else:
        return ''