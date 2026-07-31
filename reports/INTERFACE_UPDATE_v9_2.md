# Interface update v9.2

- Removed the standalone **Knowledge engine** page from the sidebar.
- Added a **Literature search** page powered by Tavily.
- Searches use advanced retrieval and are restricted to selected scholarly publishers, indexing services and preprint repositories.
- Users can choose a publication window, number of results and optional MOF-focused query expansion.
- Results show title, publisher domain, available publication date, relevance score, summary, DOI when detectable and a direct publisher link.
- Results can be exported as CSV.
- The Tavily key is read from Streamlit Secrets as `TAVILY_API_KEY`; a temporary password field is available for local use when no secret is configured.

The literature search is a retrieval aid. Bibliographic metadata and scientific claims must be verified on the original publisher page.
