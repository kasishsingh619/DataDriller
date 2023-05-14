import ast
import pydot

def generate_flowchart(code_file, output_file):
    with open(code_file, 'r') as f:
        code = f.read()

    graph = pydot.Dot(graph_type='digraph')

    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            function_name = node.name
            node_label = f"{function_name}()"
            graph.add_node(pydot.Node(function_name, label=node_label, shape='rectangle'))

            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    for arg in child.args:
                        if isinstance(arg, ast.Name):
                            arg_name = arg.id
                            if arg_name == function_name:
                                continue
                            graph.add_edge(pydot.Edge(arg_name, function_name))

    graph.write_png(output_file)

generate_flowchart('C:\\Users\\kasis\\OneDrive\\Desktop\\Twitter-Bird-main-final\\Twitter-Bird-main\\Twitter-Bird-main\\app.py', 'output.png')