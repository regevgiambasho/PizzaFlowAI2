# PizzaFlow AI

This folder contains a Streamlit prototype for the final assignment in the Generative AI course.
The public demo uses a transparent local AI decision engine and a grounded local manager assistant.
It does not require an API key, an external model, or a paid service.

## Run locally

```bash
pip install -r requirements_final.txt
streamlit run app.py
```

No Streamlit Secrets are needed. The manager assistant answers from the PizzaFlow specification
and the current simulation output using local intent matching and grounded response templates.

## Demonstration sequence

1. Open **Dashboard**.
2. Click **Run Demo Scenario**.
3. Click **Analyze & Compare**.
4. Show the AI recommendation and the FIFO comparison.
5. Ask the local manager assistant a question, for example: “איך האלגוריתם משפר את ה־SLA?”
6. Open **Method & Responsible AI** to explain the assumptions and controls.

The comparison uses the same orders and the same simulator for both strategies. The local AI engine
evaluates transparent candidate schedules; the simulator calculates profit, delivery time and on-time rate.
