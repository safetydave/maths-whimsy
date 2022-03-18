from BPTK_Py import sddsl
import networkx as nx

STOCK = 'stock'
FLOW = 'flow'
CONSTANT = 'constant'
CONVERTER = 'converter'


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


def equation_elements(equation):
    elements = []
    if hasattr(equation, 'element'):
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
    return elements


def sd_nodes(label, model):
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


def disp_eqn(eqn):
    return str(eqn).replace('model.memoize', '')


def set_eqn_attr(G, model):
    nx.set_node_attributes(G, {(STOCK, k): disp_eqn(v.equation) for k, v in model.stocks.items()}, 'eqn')
    nx.set_node_attributes(G, {(FLOW, k): disp_eqn(v.equation) for k, v in model.flows.items()}, 'eqn')
    nx.set_node_attributes(G, {(CONSTANT, k): disp_eqn(v.equation) for k, v in model.constants.items()}, 'eqn')
    nx.set_node_attributes(G, {(CONVERTER, k): disp_eqn(v.equation) for k, v in model.converters.items()}, 'eqn')


def model_graph(model):
    G = nx.DiGraph()
    # for convenient node order, and for inclusion of CONSTANT nodes
    G.add_nodes_from(sd_nodes(STOCK, model))
    G.add_nodes_from(sd_nodes(FLOW, model))
    G.add_nodes_from(sd_nodes(CONSTANT, model))
    G.add_nodes_from(sd_nodes(CONVERTER, model))
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
    G = model_graph(model)
    gpos = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=10000,
                           nodelist=sd_nodes(STOCK, model),
                           node_color='turquoise', node_shape='s')
    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=5000,
                           nodelist=sd_nodes(FLOW, model),
                           node_color='aquamarine', node_shape='D')
    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=5000,
                           nodelist=sd_nodes(CONSTANT, model),
                           node_color='lightgrey', node_shape='o')
    nx.draw_networkx_nodes(G, ax=ax, pos=gpos, node_size=5000,
                           nodelist=sd_nodes(CONVERTER, model),
                           node_color='lightgrey', node_shape='H')

    nx.draw_networkx_edges(G, ax=ax, pos=gpos, node_size=15000,
                           edge_color='grey', width=3, arrowsize=20)

    nx.draw_networkx_labels(G, ax=ax, pos=gpos, labels=node_labels(G, eqn=eqn))
