from ray.experimental.dag import (
    DAGNode,
    InputNode,
    InputAttributeNode,
    FunctionNode,
    ClassNode,
    ClassMethodNode,
)

import os
import tempfile

from ray.experimental.dag.utils import DAGNodeNameGenerator


def check_pydot_and_graphviz():
    """Check if pydot and graphviz are installed.

    pydot and graphviz are required for plotting. We check this
    during runtime rather than adding them to Ray dependencies.

    """
    try:
        import pydot
    except ImportError:
        raise ImportError(
            "pydot is required to plot DAG, " "install it with `pip install pydot`."
        )
    try:
        pydot.Dot.create(pydot.Dot())
    except (OSError, pydot.InvocationException):
        raise ImportError(
            "graphviz is required to plot DAG, "
            "download it from https://graphviz.gitlab.io/download/"
        )


def get_nodes_and_edges(dag: DAGNode):
    """Get all unique nodes and edges in the DAG.

    A basic dfs with memorization to get all unique nodes
    and edges in the DAG.
    Unique nodes will be used to generate unique names,
    while edges will be used to construct the graph.
    """

    def _dfs(node, res, visited):
        if node not in visited:
            visited.append(node)
            for child_node in node._get_all_child_nodes():
                res.append((child_node, node))
                _dfs(child_node, res, visited)

    edges = []
    nodes = []
    _dfs(dag, edges, nodes)
    return nodes, edges


def get_node_shape(node: DAGNode):
    if isinstance(node, (InputNode, InputAttributeNode)):
        return "box"
    else:
        return "oval"


def get_edge_label(src: DAGNode, dst: DAGNode):
    if isinstance(src, InputNode) and isinstance(dst, InputAttributeNode):
        return "input attribute"
    elif isinstance(src, ClassNode) and isinstance(dst, ClassMethodNode):
        return "class method"
    else:
        return ""


def dag_to_dot(dag: DAGNode):
    """Create a Dot graph from dag.

    TODO(lchu):
    1. add more Dot configs in kwargs,
    e.g. rankdir, alignment, etc.
    2. add more contents to graph,
    e.g. args, kwargs and options of each node

    """
    # Step 0: check dependencies and init graph
    check_pydot_and_graphviz()
    import pydot

    graph = pydot.Dot(rankdir="LR")

    # step 1: generate unique name for each node in dag
    nodes, edges = get_nodes_and_edges(dag)
    name_generator = DAGNodeNameGenerator()
    node_names = {}
    for node in nodes:
        node_names[node] = name_generator.get_node_name(node)

    # step 2: create graph
    for node in nodes:
        node_name = node_names[node]
        node_shape = get_node_shape(node)
        graph.add_node(pydot.Node(node_name, shape=node_shape))
    for edge in edges:
        src = edge[0]
        dst = edge[1]
        src_name = node_names[src]
        dst_name = node_names[dst]
        edge_label = get_edge_label(src, dst)
        graph.add_edge(pydot.Edge(src_name, dst_name, label=edge_label))

    return graph


def plot(dag: DAGNode, to_file=None):
    if to_file is None:
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png")
        to_file = tmp_file.name
        extension = "png"
    else:
        _, extension = os.path.splitext(to_file)
        if not extension:
            extension = "png"
        else:
            extension = extension[1:]

    graph = dag_to_dot(dag)
    graph.write(to_file, format=extension)

    # Render the image directly if running inside a Jupyter notebook
    try:
        from IPython import display

        return display.Image(filename=to_file)
    except ImportError:
        pass

    # close temp file if needed
    try:
        tmp_file.close()
    except NameError:
        pass
