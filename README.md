# Jev model router

This is the accompanying repository for the [Jev video](https://www.youtube.com/watch?v=rpQqUEmRbaA).

[![Watch the Jev video](https://img.youtube.com/vi/rpQqUEmRbaA/hqdefault.jpg)](https://ranand12.github.io/jev-model-router/)

**[Play the embedded video](https://ranand12.github.io/jev-model-router/)**.

A private Cloud Run service that exposes a text and function-tool subset of the OpenAI Chat Completions API. For `model: "auto"`, it asks Jev to choose a capability tier, then forwards the request to the corresponding Gemini model through Google Cloud's OpenAI-compatible endpoint. Callers can also select `fast`, `balanced`, or `deep` directly.

## API

- `POST /v1/chat/completions`: non-streaming text and function-tool requests
- `GET /v1/models`: router aliases
- `GET /health`: health check

The response includes `X-Router-Tier`, `X-Router-Reason`, `X-Router-Model`, and `X-Router-Elapsed-Ms` headers. If Jev is unavailable or returns an invalid or uncertain decision, the router uses the `deep` tier.

## Configuration

| Setting | Purpose |
| --- | --- |
| `OPENROUTER_API_KEY` | OpenRouter credential for the Jev Decisions API. Inject from Google Secret Manager. |
| `GOOGLE_CLOUD_PROJECT` | Google Cloud project that has Vertex AI access. |
| `GOOGLE_CLOUD_LOCATION` | Gemini endpoint location; defaults to `global`. |
| `FAST_MODEL`, `BALANCED_MODEL`, `DEEP_MODEL` | Optional model ID overrides. |
| `ROUTE_THRESHOLD` | Optional minimum selected-choice probability; defaults to `0.70`. |

Google Application Default Credentials provide the Vertex AI access token. The Cloud Run service identity needs permission to call Vertex AI. Deploy this service with Cloud Run authentication enabled, grant invocation only to intended callers, and provide the OpenRouter key through Secret Manager. The application itself does not authenticate incoming requests.

## Run locally

Install dependencies with `pip install -r requirements.txt`, configure the required credentials and project, then run `gunicorn --bind 127.0.0.1:8080 server:app`. Run offline tests with `python -m unittest -v test_router.py`.

## Scope

The endpoint supports non-streaming text and function tools. It does not implement streaming, images, audio, or the full OpenAI API. Conversation state sent to Jev goes to OpenRouter; completion requests go to Google Cloud.
