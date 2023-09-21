"""
Module for visualising BPTK system dynamics models as graphs using Network
"""

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
    """Return the label, if any, of an equation element (stock, flow, etc)"""
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
    """Return all items in model for a given label, as [(name, object)]"""
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
    """Return all nodes in the graph representation of the model as {id: (name, object)}"""
    nodes = {}
    nid = 0
    for l in LABELS:
        for item in sd_items(l, model):
            nodes[nid] = item
            nid = nid + 1
    return nodes


def sd_node_id(name, nodes):
    """Return the id of the node with the given name, where nodes is {id: (name, object)}"""
    return list(nodes.keys())[list(v[0] for v in nodes.values()).index(name)]


ALL_ARG_ATTRS = [
    # order more specific args first for complete matching of args
    # omit smooth & trend as they require specialised connections
    # ref https://bptk.transentis.com/sd-dsl/sd_dsl_functions/sd_dsl_functions.html
    ['left', 'right', 'mean', 'stddev'], # normalcdf
    ['lower_bound', 'mode', 'upper_bound'], # triangular
    ['if_', 'then_', 'else_'], # if
    ['volume', 'first_pulse', 'interval'], # pulse
    ['input_function', 'delay_duration', 'initial_value'], # delay
    ['element_1', 'element_2'], # +, -, min, max, etc
    ['lhs', 'rhs'], # and, or
    ['height', 'timestep'], # step
    ['amplitude', 'period'], # sinwave, coswave
    ['min_value', 'max_value'], # random
    ['shape', 'scale'], # gamma, pareto, weibull
    ['mean', 'scale'], # logistic
    ['mean', 'stddev'], # lognormal, normal
    ['element'], # sin, cos, etc
    ['condition'], # not
    ['operator'], # round
    ['a', 'b'], # beta
    ['n', 'p'], # binomial
    ['n', 'r'], # combinations, permutations
    ['x'], # sqrt
    ['l'], # exprnd
    ['n'], # factorial
    ['p'], # geometric, montecarlo
    ['mu'], # poisson
]


def attr_match(equation, all_arg_attrs):
    """Find the set of arguments that matches the attributes of this equation"""
    for aa in all_arg_attrs:
        if sum([hasattr(equation, a) for a in aa]) == len(aa):
            return aa
    return []


def equation_terms(equation):
    """Recurse through all elements in equation and return a list of labelled terms"""
    terms = []
    if sd_label(equation) is not None:
        terms = [equation]
    else:
        attr_terms = attr_match(equation, ALL_ARG_ATTRS)
        for a in attr_terms:
            terms.extend(equation_terms(getattr(equation, a)))
        # smooth & trend create new networks requiring specialised connections
        if isinstance(equation, sddsl.operators.Smooth):
            terms.extend([equation.smooth])
        if isinstance(equation, sddsl.operators.Trend):
            terms.extend([equation.trend])
    return terms


def disp_eqn(eqn):
    """Make an equation more readable for display"""
    return str(eqn).replace('model.memoize', '')


def sd_links(nodes):
    """Return all the links [(id1, id2)] between nodes resulting from equations in the model"""
    links = []
    for n, v in nodes.items():
        new_links = [(sd_node_id(e.name, nodes), n)
                      for e in equation_terms(v[1].equation)
                      if hasattr(e, 'name')]
        links.extend(new_links)
    return links


def has_generated_net(model, i):
    """Test if the model has a generated smooth or trend network at count i"""
    for n in model.converters:
        if n == f'bptk_{i}_input_function':
            return True
    return False


def generated_net_interface(model, graph, i):
    """Determine the external interface of generated smooth or trend network"""
    net_type = None
    net_ids = []
    inputs = []
    outputs = []
    for j, l in nx.get_node_attributes(graph, 'label').items():
        if l.startswith(f'bptk_{i}_'):
            net_ids.append(j)
            in_edges = list(graph.in_edges(nbunch=[j]))
            inputs.extend([e[0] for e in in_edges])
            if l == f'bptk_{i}_smooth' or l == f'bptk_{i}_trend':
                outputs = [e[1] for e in list(graph.out_edges(nbunch=[j]))]
                net_type = l.split('_')[-1]
    return net_type, net_ids, inputs, outputs


def collapse_generated_nets(model, graph):
    """Collapse all generated networks, preserving their interface to the rest of the model"""
    i = 1
    added = {}
    remove = []
    while has_generated_net(model, i):
        net_type, net_ids, inputs, outputs = generated_net_interface(model, graph, i)
        new_id = graph.number_of_nodes()
        graph.add_node(new_id)
        added[new_id] = f'{net_type} {i}'
        graph.add_edges_from([(j, new_id) for j in inputs if j not in net_ids])
        graph.add_edges_from([(new_id, j) for j in outputs if j not in net_ids])
        remove.extend(net_ids)
        i = i + 1
    graph.remove_nodes_from(remove)
    return added


def model_graph(model, collapse=False):
    """Create a NetworkX DiGraph from the BPTK model, optionally collapse generated [sub]networks"""
    G = nx.DiGraph()
    nodes = sd_nodes(model)
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G, {n: e[1].name for n, e in nodes.items()}, 'label')
    nx.set_node_attributes(G, {n: (sd_label(e[1]) + ': ' + disp_eqn(e[1].equation)) for n, e in nodes.items()}, 'title')
    nx.set_node_attributes(G, {n: COLORS[sd_label(e[1])] for n, e in nodes.items()}, 'color')
    nx.set_node_attributes(G, {n: SHAPES[sd_label(e[1])] for n, e in nodes.items()}, 'shape')
    G.add_edges_from(sd_links(nodes))

    if collapse:
        added = collapse_generated_nets(model, G)
        nx.set_node_attributes(G, added, 'label')
        nx.set_node_attributes(G, {n: f'collapsed {l} network' for n, l in added.items()}, 'title')
        nx.set_node_attributes(G, {n: COLORS[STOCK] for n in added}, 'color')
        nx.set_node_attributes(G, {n: 'triangleDown' for n in added}, 'shape')

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
                      for e in equation_terms(v.equation)
                      if hasattr(e, 'name')])
    return edges


def set_eqn_attr(G, model):
    nx.set_node_attributes(G, {(STOCK, k): disp_eqn(v.equation) for k, v in model.stocks.items()}, 'eqn')
    nx.set_node_attributes(G, {(FLOW, k): disp_eqn(v.equation) for k, v in model.flows.items()}, 'eqn')
    nx.set_node_attributes(G, {(CONSTANT, k): disp_eqn(v.equation) for k, v in model.constants.items()}, 'eqn')
    nx.set_node_attributes(G, {(CONVERTER, k): disp_eqn(v.equation) for k, v in model.converters.items()}, 'eqn')


def model_graph_labels(model):
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
    G = model_graph_labels(model)
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
