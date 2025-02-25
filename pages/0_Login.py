import streamlit as st
from utils.auth import AuthManager

# Function to inject local CSS styles
def local_css(css_text):
    st.markdown(f'<style>{css_text}</style>', unsafe_allow_html=True)

# Custom CSS for modern look
custom_css = """
/* Overall page background */
body {
    background: linear-gradient(to right, #1e3c72, #2a5298);
    font-family: 'Roboto', sans-serif;
}

/* Centering the login card */
.login-container {
    max-width: 400px;
    margin: 5% auto;
    background: #fff;
    padding: 2rem;
    border-radius: 10px;
    box-shadow: 0px 0px 20px rgba(0, 0, 0, 0.1);
}

/* Styling the form elements */
.login-container .stTextInput input,
.login-container .stTextInput label,
.login-container .stSelectbox select {
    font-size: 1rem;
}

/* Custom button styles */
.stButton>button {
    background-color: #2a5298;
    color: white;
    border-radius: 5px;
    border: none;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    transition: background 0.3s ease;
}
.stButton>button:hover {
    background-color: #1e3c72;
}

/* Optional: style the checkbox */
.stCheckbox>div {
    font-size: 0.9rem;
    color: #333;
}
"""

def app():
    # Inject custom CSS
    local_css(custom_css)

    # Wrap the content in a container div for styling
    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.title("🔒 Poultry Weight Predictor Login")

    # Initialize auth manager
    auth = AuthManager()

    # Login form
    with st.form(key='login_form'):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label="Login")

        if submit_button:
            user_info = auth.authenticate(username, password)
            if user_info:
                st.session_state['authenticated'] = True
                st.session_state['user_info'] = user_info
                st.success(f"Welcome, {username}!")
                st.session_state['current_page'] = "1_Data_Upload"
                st.rerun()
            else:
                st.error("Invalid username or password")

    # Registration option (optional, restrict to admin)
    if st.checkbox("Register New User (Admin Only)"):
        with st.form(key='register_form'):
            new_username = st.text_input("New Username")
            new_password = st.text_input("New Password", type="password")
            role = st.selectbox("Role", ["user", "admin"])
            register_button = st.form_submit_button(label="Register")

            if register_button:
                if 'user_info' in st.session_state and st.session_state['user_info']['role'] == 'admin':
                    if auth.add_user(new_username, new_password, role):
                        st.success(f"User {new_username} registered successfully!")
                    else:
                        st.error("Username already exists!")
                else:
                    st.error("Only admins can register new users!")
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    app()
