# Security Notice

The dashboard backend is designed for **local development and demonstration purposes only**.

By default, the backend allows CORS origins from `http://localhost:8000`. 
If you intend to deploy this system or expose it beyond localhost, **you must implement proper authentication** (e.g., JWT, API keys, or OAuth).

Do not expose the raw FastAPI backend over the internet without securing the endpoints, as it allows arbitrary video uploads and unauthenticated access to the processing pipeline.

To change allowed CORS origins, set the environment variable:
`TRAFFIC_ALLOWED_ORIGINS="https://your-domain.com"`
