import json
from agent.graph import build_graph

g = build_graph()
result = g.invoke(
    {"raw_input": "2401.12345"},
    config={"configurable": {"thread_id": "test_summarize_final"}},
)
briefing = result.get("briefing")
print("url:", briefing.get("url"))
print("meta:", briefing.get("meta"))

with open("examples/briefing_2401_12345.json") as f:
    saved = json.load(f)
print("saved file meta:", saved.get("meta"))
