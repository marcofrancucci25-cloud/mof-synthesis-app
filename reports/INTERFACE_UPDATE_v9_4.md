# Interface update v9.4

- Removed all Tavily API-key input and deployment-secret checks from the user interface.
- Added one clearly marked Tavily key constant in `src/literature.py`.
- Literature searches now run directly from the search form.
- Retained trusted-domain filtering, recent-publication window, MOF-context option and CSV export.
- Predictive model and validation artifacts are unchanged.
