## AI features

RistoAI Backend includes three AI capabilities.

### 1. AI insights (pluggable strategies)

Insights run through a strategy layer, so new insight types can be added without touching the rest of the code. Results are stored in the `ai_insights` collection.

| Strategy | Purpose |
|---|---|
| Demand forecast | Predicts demand for menu items |
| Menu optimization | Suggests menu improvements |
| Recommendations | Suggests dishes and pairings |
| Waste reduction | Highlights where waste can be cut |

Code: `app/services/strategies/`

### 2. AI chat assistant

- OpenAI-powered chat for restaurant staff
- Chat history stored per user
- Translation of the latest 40 messages
- Clear error handling when the OpenAI key is missing or the call fails

### 3. Voice transcription

- Speech-to-text with the OpenAI Whisper API
- The user's app language is passed explicitly, to improve accuracy
