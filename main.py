"""
CLIENT DEFAULT PAYMENT RISK WEB APPLICATION
Shows clients payment default risk
Allows user to customize impacting variables to see impact on the calculated risk
Provides comparison with other clients (from the training group)
"""

import numpy as np
import shap
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import requests
import lightgbm


# -------- GENERAL VAL INST    --------
# url_api = "http://localhost:8000/predict/"  # FastAPI url local
url_api = 'https://oc7-fastapi.azurewebsites.net/predict/'  # FastAPI url on same server the app is running

# --------   FUNCTIONS   --------

# >> Prepare cached environment

# Dataset
@st.cache_data
def load_dataset():
    # Import X_test et y_test
    X_test = pd.read_csv('data_test_clean/X_test_sample.csv')
    y_test = pd.read_csv('data_test_clean/y_test_sample.csv')
    data = pd.concat((X_test, y_test), axis=1)
    return X_test, y_test, data


# # Model (Faster for the app treatment)
# @st.cache_data
# def get_model():
#     return mlflow.lightgbm.load_model(model_uri='models:/LightGBM/latest')


# Stat values
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
def get_shap_values():
    return explainer(X_test.drop('SK_ID_CURR', axis=1))[:, :, 1]


# Get Clients info
def clients_real(id_client):
    df_client = data.loc[data.SK_ID_CURR == id_client].drop(['SK_ID_CURR', 'true'], axis=1)
    true = data.loc[data.SK_ID_CURR == id_client, 'true'].values[0]
    return df_client, true


def get_prediction(df_client_tolist):
    response = requests.post(url_api, json={'data': df_client_tolist})
    if response.status_code == 200:
        result = response.json()
        pred = result['pred']
        proba = result['proba']
        return pred, proba
    else:
        st.error('Error at api call')
        return None


def shap_explanation(df_client):
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


def get_clients_info(df_client):
    # Call api predictions for selected client
    pred, proba = get_prediction(df_client.values.tolist())
    # Shap explanation
    client_score, client_data = shap_explanation(df_client)
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


def calc_risk_level(client_score):
    risk_level = 'Medium'
    risk_color = 'grey'
    if client_score < median_sc0:
        risk_level = 'Low'
        risk_color = c1
    elif client_score > median_sc1:
        risk_level = 'High'
        risk_color = c2
    return risk_level, risk_color


def _update_page_status(status):
    st.session_state.status = status


def create_slider(sd, slider_key):
    return st.sidebar.slider(label=sd['label'], min_value=sd['min'], max_value=sd['max'],
                             value=sd['val'], step=sd['step'], key=slider_key)


def highlight_scores_values(val):
    if val > 0:
        return f'background-color: {c2}'
    else:
        return ''


# --------   GLOBAL ENVIRONMENT   --------

X_test, y_test, data = load_dataset()

# Get latest registered model
# model = get_model()

# Create ShapValues Explainer
explainer = get_shap_explainer()
shap_values = get_shap_values()

# Get Globals info
mini_maxi_features = get_general_features_values()
sc0, sc1, median_sc0, median_sc1 = get_median_scores()

# --------   WEB APP   --------

# HEADER -----------

# >> Style
c1 = '#125fc4'
c2 = '#f5a142'

st.markdown('''
<style>
    #simulation{background: #E6E6E6; padding: 15px; border-radius: 50px; margin-top:20px;}
</style>
''', unsafe_allow_html=True)



# SIDEBAR -----------

# Init session state
if 'status' not in st.session_state:
    st.session_state['status'] = 'real'

st.sidebar.write("# Client risk Simulator")

# >> Client ID selector

# Get first client of the list
url_idx = 0
# Else, if exists, get url index client
if st.experimental_get_query_params():
    url_idx = int(st.experimental_get_query_params()['idx'][0])

# Select client
client_id = st.sidebar.selectbox('Select or type your client ID', X_test['SK_ID_CURR'], index=url_idx)

# Get current client id index
client_idx = X_test.loc[X_test['SK_ID_CURR'] == client_id].index[0]
# If different (ie client change), empty session state variables and set new url parameter
if client_idx != url_idx:
    for key in st.session_state.keys():
        del st.session_state[key]
    _update_page_status('real')
    st.experimental_set_query_params(idx=client_idx)

# Get client info and
df_client, true = clients_real(client_id)

# Create simulation client dataframe
df_client_sim = df_client.copy()

# Get clients real predictions
real_pred, real_proba, real_client_score, real_client_data = get_clients_info(df_client)

# Create sliders dict (sliders initial values for chosen variables)
sliders = create_sliders_variables(real_client_data, mini_maxi_features)


# >> Sliders section
st.sidebar.write('''
#### Play with impacting features
Those features increase your clients risk estimation. Change their values to see if the risk changes.
''')

# --------------------
# Reset Button (resets sliders to initial values)
if st.sidebar.button('Reset client values'):
    for key, value in sliders.items():
        st.session_state[key] = value['val']
    _update_page_status('real')
# --------------------

# Check sessionstate values
# st.sidebar.write(st.session_state)

# >> Create Sliders
slider1 = create_slider(sliders['slider1'], 'slider1')
slider2 = create_slider(sliders['slider2'], 'slider2')
slider3 = create_slider(sliders['slider3'], 'slider3')

# Change simulation dataframe for further previsions
df_client_sim[sliders['slider1']['label']] = st.session_state['slider1']
df_client_sim[sliders['slider2']['label']] = st.session_state['slider2']
df_client_sim[sliders['slider3']['label']] = st.session_state['slider3']

# Check it values different from real ones and change page status
if (df_client_sim != df_client).any().any():
    _update_page_status('simulation')
else:
    _update_page_status('real')


# Check dataframes values and session state variables
# df_client[[sliders['slider1']['label'], sliders['slider2']['label'], sliders['slider3']['label']]]
# df_client_sim[[sliders['slider1']['label'], sliders['slider2']['label'], sliders['slider3']['label']]]
# st.write(st.session_state)


# PAGE -----------

# Recalculate predictions depending en page session (real or simulation)
if st.session_state.status == 'real':
    df_client_to_use = df_client
else:
    df_client_to_use = df_client_sim

pred, proba, client_score, client_data = get_clients_info(df_client_to_use)
risk_level, risk_color = calc_risk_level(client_score)

# >> Set layout
col1, col2 = st.columns([4, 5], gap='large')

# ----- Column 1 : Clients risk
# If simulation, create real data comparison
real_risk_mess = real_proba_mess = real_score_mess = real_level_mess = ''
bf = ' ('
af = ')'
rsk = 'NO'
if real_pred == 1:
    rsk = 'YES'
rsklvl, rskc = calc_risk_level(real_client_score)
if st.session_state.status == 'simulation':
    real_risk_mess = bf + rsk + af
    real_proba_mess = bf + str(np.round(real_proba * 100, 0)) + '%' + af
    real_score_mess = bf + str(real_client_score) + af
    real_level_mess = bf + rsklvl + af

# Show calculated data
col1.markdown('#### Is client {} at risk?'.format(client_id))

# If simulation, show a message
if st.session_state.status == 'simulation':
    col1.markdown('<div style="background :#E6E6E6; font-size: 1.2em">*** Simulation mode ***</div>',
                  unsafe_allow_html=True)

col1.divider()
if pred == 1:
    risk = '\u26A0\ufe0f **YES**'
else:
    risk = '**NO**'
col1.markdown('Presents a default payment risk : {}{}'.format(risk, real_risk_mess))
col1.markdown('Predicted risk probability : **{}%**{}'.format(np.round(proba * 100, 0), real_proba_mess))
col1.markdown('Client Score : **{}**{}'.format(client_score, real_score_mess))
col1.markdown('Level of risk : <span style="color:{}">**{}**</span>{}'.format(risk_color, risk_level, real_level_mess),
              unsafe_allow_html=True)

# col1.markdown('TRUE = {}'.format(y_test.iloc[client_idx][0]))

# ----- Column 2 : scores density graph

col2.write('#### Client score vs other clients scores')
graph = plt.figure()
sns.kdeplot(sc0, palette=[c1], fill=True, label='No risk clients')
sns.kdeplot(sc1, palette=[c2], fill=True, label='At risk clients')
plt.axvline(x=client_score, color=risk_color, label='Client Score', linestyle='--', linewidth=3)
plt.xlabel('Score')
plt.ylabel('Scores distribution')
plt.legend()
col2.pyplot(graph)

st.divider()

# ----- Body

# >> Score Decomposition
st.write('''
#### Score decomposition
Impact of clients features, from higher to lower impact.
A positive score means the feature increases the payment default risk.
''')

# Clients data highlited with color on positive score values (ie values increasing the risk)
st.dataframe(client_data.style.applymap(highlight_scores_values, subset=['feature_score']))

st.divider()

# 2 Columns layout
col3, col4 = st.columns([4,5], gap='large')

# >> Global feature importance
col3.markdown('''
#### Global Feature Importance
Features general behavior <br />
<span style="font-size:0.7em">*left = lowers the risk, right = increases the risk*<br />
*red/dark = when feature value is higher, blue/light = when value is lower*</span>
''', unsafe_allow_html=True)

plt.figure(figsize=(10,20))
shap_global = shap.summary_plot(shap_values, X_test.drop('SK_ID_CURR', axis=1))
plt.savefig('refs/shap_summary_plot.png')
col3.image('refs/shap_summary_plot.png')

# >> Global feature importance
col4.markdown('''
#### Client comparison
Current clients position on 2 important risky features versus other clients<br />
<span style="font-size:0.7em">*Bigger point for current client*</span>
''', unsafe_allow_html=True)
# ft1 = client_data.reset_index().loc[0,'feature']
# ft2 = client_data.reset_index().loc[1,'feature']
ft1 = 'EXT_SOURCE_2'
ft2 = 'EXT_SOURCE_3'

df_comparison = data[[ft1, ft2, 'true']]
df_comparison_client = pd.DataFrame(df_comparison.iloc[client_idx]).T
comparison = plt.figure()
# sns.scatterplot(df_comparison, x=ft1, y=ft2, hue='true')
sns.scatterplot(df_comparison, x=ft1, y=ft2, hue='true', style='true')
sns.scatterplot(df_comparison_client, x=ft1, y=ft2, s=200, label='Current', color='black')
plt.legend(title='Risky clients', bbox_to_anchor=(1.05, 1))
col4.pyplot(comparison)







