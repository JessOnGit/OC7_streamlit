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
import utils


# -------- GENERAL VAL INST    --------
# url_api = "http://localhost:8000/predict/"  # FastAPI url local
url_api = 'https://oc7-fastapi.azurewebsites.net/predict/'  # FastAPI url on same server the app is running


# --------   GLOBAL ENVIRONMENT   --------

X_test, y_test, data = utils.load_dataset()

# Get latest registered model
# model = get_model()

# Create ShapValues Explainer
explainer = utils.get_shap_explainer()
shap_values = utils.get_shap_values(explainer, X_test)

# Get Globals info
mini_maxi_features = utils.get_general_features_values()
sc0, sc1, median_sc0, median_sc1 = utils.get_median_scores()

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
    utils.update_page_status('real')
    st.experimental_set_query_params(idx=client_idx)

# Get client info and
df_client, true = utils.clients_real(client_id, data)

# Create simulation client dataframe
df_client_sim = df_client.copy()

# Get clients real predictions
real_pred, real_proba, real_client_score, real_client_data = utils.get_clients_info(df_client, url_api, explainer)

# Create sliders dict (sliders initial values for chosen variables)
sliders = utils.create_sliders_variables(real_client_data, mini_maxi_features)


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
    utils.update_page_status('real')
# --------------------

# Check sessionstate values
# st.sidebar.write(st.session_state)

# >> Create Sliders
slider1 = utils.create_slider(sliders['slider1'], 'slider1')
slider2 = utils.create_slider(sliders['slider2'], 'slider2')
slider3 = utils.create_slider(sliders['slider3'], 'slider3')

# Change simulation dataframe for further previsions
df_client_sim[sliders['slider1']['label']] = st.session_state['slider1']
df_client_sim[sliders['slider2']['label']] = st.session_state['slider2']
df_client_sim[sliders['slider3']['label']] = st.session_state['slider3']

# Check it values different from real ones and change page status
if (df_client_sim != df_client).any().any():
    utils.update_page_status('simulation')
else:
    utils.update_page_status('real')


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

pred, proba, client_score, client_data = utils.get_clients_info(df_client_to_use, url_api, explainer)
risk_level, risk_color = utils.calc_risk_level(client_score, median_sc0, median_sc1, c1, c2)

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
rsklvl, rskc = utils.calc_risk_level(real_client_score, median_sc0, median_sc1, c1, c2)
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
st.dataframe(client_data.style.applymap(lambda x: utils.highlight_scores_values(x, c2), subset=['feature_score']))

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