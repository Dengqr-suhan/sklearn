import pydot

(graph,) = pydot.graph_from_dot_file(r"C:\Users\18873\Desktop\python\python从入门到精通\机器学习\wine.dot")
graph.write_png(r"C:\Users\18873\Desktop\python\python从入门到精通\机器学习\wine.png")
