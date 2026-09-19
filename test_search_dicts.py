from agent.graph import build_graph

g = build_graph()
result = g.invoke(
    {"raw_input": "recent work on KV-cache compression for LLMs"},
    config={"configurable": {"thread_id": "test_regression_fix"}},
)
print("paper title:", (result.get("paper") or {}).get("title"))
print("paper url:", (result.get("paper") or {}).get("url"))
