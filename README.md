## AI features

RistoAI Backend includes three AI capabilities.

### 1. AI insights (pluggable strategies)

Insights run through a strategy layer (Strategy pattern), so new insight types can be added without touching the rest of the code. The design also allows ML-backed implementations to be swapped in later. Results are stored in the `ai_insights` collection.

| Strategy | Purpose |
|---|---|
| Demand forecast | Demand forecast insight |
| Menu optimization | Menu optimization insight |
| Recommendations | Recommendation insight |
| Waste reduction | Waste reduction insight |

Code: `app/services/strategies/`

### 2. AI chat assistant

- OpenAI-powered chat assistant
- Full conversation history loads
- Translation covers the latest 40 messages only, to avoid timeouts on Vercel serverless functions
- Missing API key or OpenAI failures return explicit errors instead of a generic fallback

### 3. Voice transcription

- Speech-to-text with the OpenAI Whisper API
- The app's selected language is passed explicitly to Whisper to reduce misinterpretation
- Transcripts are not auto-translated, so the user's exact phrasing is kept

<!-- [optional: add a sample request/response for one insight, as a code block] -->
