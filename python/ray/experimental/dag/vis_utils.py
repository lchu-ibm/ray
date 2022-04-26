from typing import Dict

from ray.experimental.dag import DAGNode, InputNode, InputAtrributeNode, ClassMethodNode

import os
import tempfile


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

    # Step 1: generate unique name for each node in dag
    nodes, edges = get_nodes_and_edges(dag)
    name_generator = DAGNodeNameGenerator()
    node_names = {}
    for node in nodes:
        node_names[node] = name_generator.get_node_name(node)

    # Step 2: create graph with all the edges
    for edge in edges:
        graph.add_edge(pydot.Edge(node_names[edge[0]], node_names[edge[1]]))

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


class DAGNodeNameGenerator(object):
    """
    Generate unique suffix for each given Node in the DAG.
    Apply monotonic increasing id suffix for duplicated names.
    """

    def __init__(self):
        self.name_to_suffix: Dict[str, int] = dict()

    def get_node_name(self, node: DAGNode):

        if isinstance(node, InputNode):
            node_name = "input_node"
        elif isinstance(node, InputAtrributeNode):
            node_name = "input_attributed_node"
        elif isinstance(node, ClassMethodNode):
            node_name = node.get_options().get("name", None) or node._method_name
        else:
            node_name = node.get_options().get("name", None) or node._body.__name__

        if node_name not in self.name_to_suffix:
            self.name_to_suffix[node_name] = 0
            return node_name
        else:
            self.name_to_suffix[node_name] += 1
            suffix_num = self.name_to_suffix[node_name]

            return f"{node_name}_{suffix_num}"

    def reset(self):
        self.name_to_suffix = dict()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.reset()
