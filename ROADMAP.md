# Roadmap

Planned work for PDF LLM Chat Assistant. Items link to GitHub issues where one exists.

## Near term
- **Tests and CI**: unit tests for file parsing and API routes, run on every push.
- **Modularize the code**: split `pdf_chat_app.py` into app, OpenRouter client, file parsing, and frontend template.
- **Safer configuration**: `.env` support and no API key placeholder in source.

## Next
- **Retrieval for long documents**: chunk and embed uploads so large PDFs and multi-file sets fit within model context limits, with page-level citations in answers.
- **Pluggable model backends**: support OpenAI-compatible endpoints alongside OpenRouter.
- **Per-session document storage**: replace the global in-memory store so multiple users do not share documents.

## Later
- OCR for scanned PDFs.
- Evaluation set of sample documents and questions to compare models and catch regressions.
- Docker image and one-command deployment.
- Export chat transcripts.
