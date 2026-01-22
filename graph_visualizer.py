from graph import app

graph = app.get_graph()

mermaid = graph.draw_mermaid(with_styles=False)
print(mermaid)
