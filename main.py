import streamlit as st
from config.settings import APP_NAME, APP_ICON, LAYOUT
from utils.auth import check_authentication, logout

# Configure the Streamlit page
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout=LAYOUT
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = "0_Login"

# Page navigation dictionary
PAGES = {
    "0_Login": "pages/0_Login.py",
    "1_Data_Upload": "pages/1_Data_Upload.py",
    "2_Data_Analysis": "pages/2_Data_Analysis.py",
    "3_Model_Training": "pages/3_Model_Training.py",
    "4_Predictions": "pages/4_Predictions.py",
    "5_Model_Comparison": "pages/5_Model_Comparison.py",
    "6_About": "pages/6_About.py"
}

# Sidebar navigation (only show after login)
if 'authenticated' in st.session_state and st.session_state['authenticated']:
    st.sidebar.title(f"Welcome, {st.session_state['user_info']['username']}")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Navigate",
        options=list(PAGES.keys())[1:],  # Exclude login page
        format_func=lambda x: x.split('_', 1)[1].replace('_', ' ')
    )
    st.session_state['current_page'] = page
    
    # Logout button
    if st.sidebar.button("Logout"):
        logout()

# Load selected page
def load_page(page_name):
    if page_name != "0_Login":
        check_authentication()
    
    # Import and run the page
    page_module = __import__(PAGES[page_name].replace('/', '.').replace('.py', ''), fromlist=['app'])
    page_module.app()

# Execute current page
load_page(st.session_state['current_page'])

# Main page header and content (only show if authenticated)
if 'authenticated' in st.session_state and st.session_state['authenticated']:
    st.title("🐔 Poultry Weight Predictor")

    # Welcome message
    st.markdown("""
    ## Welcome to the Poultry Weight Predictor

    This application helps you predict poultry weight based on environmental and feeding data. 
    You can:

    1. Upload and analyze your poultry data
    2. Train machine learning models
    3. Make predictions on new data
    4. Visualize results and insights

    ### Getting Started

    Use the sidebar to navigate through different sections of the app:

    - **Data Upload**: Upload and preview your data
    - **Data Analysis**: Explore your data with visualizations
    - **Model Training**: Train and evaluate prediction models
    - **Predictions**: Make predictions on new data

    ### Required Data Format

    Your CSV file should include the following columns:
    - Internal Temperature (Int Temp)
    - Internal Humidity (Int Humidity)
    - Air Temperature (Air Temp)
    - Wind Speed
    - Feed Intake
    - Weight (target variable)
    """)

    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit")