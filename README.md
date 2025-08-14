# FreeTitle AI - Multi-Agent Video Production

Fully automated AI video production system powered by the Gemini 3 model family. One sentence in, cinematic trailer out.

## Features

- **End-to-End Production** - Automated pipeline mirroring real filmmaking: pre-production (script, characters, storyboard), production (video, audio), and post-production (editing)
- **Intelligent Orchestration** - Gemini reasons about agent execution, parallel grouping, and result evaluation at each step
- **Closed-Loop Quality** - Generated assets are visually inspected by Gemini and regenerated when needed
- **Real-Time Streaming** - Live WebSocket updates throughout the production process

## Tech Stack

- **Gemini 3 Pro** - Reasoning, orchestration, script writing, editing decisions
- **Gemini 3 Pro Image** - Characters, storyboard frames, props, environments
- **Veo 3.1** - Image-to-video generation
- **FastAPI + LangGraph** - Backend orchestration
- **Redis + GCS** - Session state and asset storage

## Quick Start

```bash
pip install -r requirements.txt
python -m src.api.main  # http://localhost:8080
```

Requires a `.env` with `GOOGLE_AI_API_KEY`, `GCP_PROJECT_ID`, `REDIS_URL`, `BUCKET_NAME`, and `SESSION_BUCKET_NAME`.

## License

Copyright (c) 2025 FreeTitle, Inc. All rights reserved. See [LICENSE](LICENSE) for details.
