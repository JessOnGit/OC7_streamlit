# OC7_streamlit
## General Description
App allowing to see client application default payment risk.  
https://oc7-streamlit.azurewebsites.net/
## Main screen
Selection of client by ID  
Results shown on main page :  
  - Predicted risk (at risk or not)
  - Clients risk score in comparison of other clients scores (risky or not risky clients)
  - Detail of feature importance for risk score calculation
  - General model feature importance fo all clients
  - Position of current client values for two main features in regard of others clients positions
## Advanced Functionnalities
Possibility to simulate current clients risk by changing the value of the 3 main features involved in current clients risk augmentation.
## Files
### Folders
- refs : reference files used for global data comparison (generated from train files)
- data_test_clean : sample of X and y (target) test set (after 80/20 split of train set)
- github/workflows : github separated actions (unit testing, launched on push + docker build and send to Azure Container Registry, manually launched)
### Files
- main.py : app code
- utils.py : functions used in main.py
- startup.sh : for local startup
- uni_tests : unit tests code. 3 unit tests (one for folder contents check, one for data presence and format check, one for function check)
- Dockerfile : for docker container build
