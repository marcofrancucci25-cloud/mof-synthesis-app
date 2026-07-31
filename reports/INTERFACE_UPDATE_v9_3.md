# Interface update v9.3

- Tavily credentials are now read only from `st.secrets`.
- The API-key input field has been removed from the public interface.
- A local `.streamlit/secrets.toml` is included for private/local use and is excluded by `.gitignore`.
- Streamlit Community Cloud still requires `TAVILY_API_KEY` to be entered in **App settings → Secrets**, because ignored files are not deployed from GitHub.
