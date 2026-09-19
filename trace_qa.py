from agent.graph import build_graph

g = build_graph()
thread = {"configurable": {"thread_id": "trace_2401"}}
g.invoke({"raw_input": "2401.12345"}, config=thread)

for q in ["What method does this paper propose?",
          "What datasets or experiments did they use?"]:
    print("\n### INVOKE:", q)
    for update in g.stream({"question": q}, config=thread, stream_mode="updates"):
        for node, out in update.items():
            n = len((out or {}).get("messages", []))
            print(f"  node={node} returned_messages={n}")
    print("  total in state:", len(g.get_state(thread).values["messages"]))