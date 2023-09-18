from BPTK_Py import sddsl
import networkx as nx

STOCK = 'stock'
FLOW = 'flow'
CONSTANT = 'constant'
CONVERTER = 'converter'
LABELS = [STOCK, FLOW, CONSTANT, CONVERTER]

COLORS = {STOCK: '#40E0D0', FLOW: '#7FFFD4', CONSTANT: '#D3D3D3', CONVERTER: '#D3D3D3'}
SHAPES = {STOCK: 'square', FLOW: 'diamond', CONSTANT: 'dot', CONVERTER: 'triangle'}


##
##  Interactive visusalisation and shared methods
##


def sd_label(element):
    t = None
    if type(element) is sddsl.stock.Stock:
        t = STOCK
    elif type(element) is sddsl.flow.Flow:
        t = FLOW
    elif type(element) is sddsl.constant.Constant:
        t = CONSTANT
    elif type(element) is sddsl.converter.Converter:
        t = CONVERTER
    return t


def sd_items(label, model):
    items = []
    if label == STOCK:
        items = model.stocks.items()
    elif label == FLOW:
        items = model.flows.items()
    elif label == CONSTANT:
        items = model.constants.items()
    elif label == CONVERTER:
        items = model.converters.items()
    return items


def sd_nodes(model):
    nodes = {}
    nid = 0
    for l in LABELS:
        for item in sd_items(l, model):
            nodes[nid] = item
            nid = nid + 1
    return nodes


def sd_node_id(name, nodes):
    return list(nodes.keys())[list(v[0] for v in nodes.values()).index(name)]


def equation_elements(equation):
    elements = []
    if sd_label(equation) is not None:
        elements = [equation]
    elif hasattr(equation, 'element'):
        elements = [equation.element]
    elif hasattr(equation, 'element_1'):
        root_elements = [equation.element_1, equation.element_2]
        left_subtree = equation_elements(equation.element_1)
        right_subtree = equation_elements(equation.element_2)
        elements = root_elements + left_subtree + right_subtree
    elif hasattr(equation, 'if_'):
        if_elements = equation_elements(equation.if_)
        then_elements = equation_elements(equation.then_)
        else_elements = equation_elements(equation.else_)
        elements = if_elements + then_elements + else_elements
    # todo handle more types
    # delay function has input_function
    return elements


def disp_eqn(eqn):
    return str(eqn).replace('model.memoize', '')


def sd_links(nodes):
    links = []
    for n, v in nodes.items():
        new_links = [(sd_node_id(e.name, nodes), n)
                      for e in equation_elements(v[1].equation)
                      if hasattr(e, 'name')]
        links.extend(new_links)
    return links


def model_graph(model):
    G = nx.DiGraph()
    nodes = sd_nodes(model)
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G, {n: e[1].name for n, e in nodes.items()}, 'label')
    nx.set_node_attributes(G, {n: (sd_label(e[1]) + ': ' + disp_eqn(e[1].equation)) for n, e in nodes.items()}, 'title')
    #nx.set_node_attributes(G, {n: sd_label(e[1]) for n, e in nodes.items()}, 'group')
    nx.set_node_attributes(G, {n: COLORS[sd_label(e[1])] for n, e in nodes.items()}, 'color')
    nx.set_node_attributes(G, {n: SHAPES[sd_label(e[1])] for n, e in nodes.items()}, 'shape')
    G.add_edges_from(sd_links(nodes))
    return G


##
##  Static visusalisation methods
##  Somewhat deprecated
##


def sd_node_keys(label, model):
    keys = []
    if label == STOCK:
        keys = model.stocks.keys()
    elif label == FLOW:
        keys = model.flows.keys()
    elif label == CONSTANT:
        keys = model.constants.keys()
    elif label == CONVERTER:
        keys = model.converters.keys()
    return [(label, k) for k in keys]


def sd_edges(label, items):
    edges = []
    for k, v in items:
        edges.extend([((sd_label(e), e.name), (label, k))
                      for e in equation_elements(v.equation)
                      if hasattr(e, 'name')])
    return edges


def set_eqn_attr(G, model):
    nx.set_node_attributes(G, {(STOCK, k): disp_eqn(v.equation) for k, v in model.stocks.items()}, 'eqn')
    nx.set_node_attributes(G, {(FLOW, k): disp_eqn(v.equation) for k, v in model.flows.items()}, 'eqn')
    nx.set_node_attributes(G, {(CONSTANT, k): disp_eqn(v.equation) for k, v in model.constants.items()}, 'eqn')
    nx.set_node_attributes(G, {(CONVERTER, k): disp_eqn(v.equation) for k, v in model.converters.items()}, 'eqn')


def model_graph_elements(model):
    G = nx.DiGraph()
    # for convenient node order, and for inclusion of CONSTANT nodes
    G.add_nodes_from(sd_node_keys(STOCK, model))
    G.add_nodes_from(sd_node_keys(FLOW, model))
    G.add_nodes_from(sd_node_keys(CONSTANT, model))
    G.add_nodes_from(sd_node_keys(CONVERTER, model))
    # otherwise, we could just rely on nodes created by edges
    G.add_edges_from(sd_edges(STOCK, model.stocks.items()))
    G.add_edges_from(sd_edges(FLOW, model.flows.items()))
    G.add_edges_from(sd_edges(CONVERTER, model.converters.items()))
    # add equation attributes
    set_eqn_attr(G, model)
    return G


def node_labels(G, eqn=True):
    return {n: f'{n}' + ('' if not eqn else f'\n\n{G.nodes[n]["eqn"]}') for n in G.nodes}


def draw_model_graph(model, ax=None, eqn=True):
    G = model_graph_elements(model)
    gpos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=8000,
                           nodelist=sd_node_keys(STOCK, model),
                           node_color='turquoise', node_shape='s')
    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=4000,
                           nodelist=sd_node_keys(FLOW, model),
                           node_color='aquamarine', node_shape='D')
    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=4000,
                           nodelist=sd_node_keys(CONSTANT, model),
                           node_color='lightgrey', node_shape='o')
    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=4000,
                           nodelist=sd_node_keys(CONVERTER, model),
                           node_color='lightgrey', node_shape='^')

    nx.draw_networkx_edges(G, ax=ax, pos=gpos, node_size=12000,
                           edge_color='grey', width=3, arrowsize=20)

    nx.draw_networkx_labels(G, ax=ax, pos=gpos, labels=node_labels(G, eqn=eqn))
