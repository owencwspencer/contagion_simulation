import networkx as nx
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import copy
import os

# --- Configuration: Define base directories ---
NETWORK_DIR = '../networks/unweighted'
RESULT_DIR = '../result_small_world/unweighted'

# --- Graph Loading and Preprocessing ---
def load_graph(file_path):
    """
    Load a graph from a GraphML file and preprocess node labels.
    """
    G = nx.read_graphml(file_path)
    mapping = {node: int(float(node)) for node in G.nodes()}
    G = nx.relabel_nodes(G, mapping)
    return G

# --- Contagion Transfer Function (Simple & Complex) ---
def perform_contagion_transfers_unweighted_version(graph, infected_list, mode='simple', threshold_complex=2, beta_simple=0.5, prob_complex=-1):
    """
    Perform one step of contagion process (simple or complex) on the given graph.
    """
    new_infection_marker = False
    new_infected_nodes = []
    all_nodes = list(graph.nodes())
    n_complex_infection = 0
    n_simple_infection = 0

    for node in all_nodes:
        if node in infected_list:
            continue

        infected_neighbors_count = sum(neighbor in infected_list for neighbor in graph.neighbors(node))

        if mode == 'simple':
            infection_probability = 1 - (1 - beta_simple)**infected_neighbors_count
            if random.uniform(0, 1) <= infection_probability:
                new_infection_marker = True
                new_infected_nodes.append(node)

        elif mode == 'complex':
            if infected_neighbors_count >= threshold_complex:
                new_infection_marker = True
                new_infected_nodes.append(node)
                n_complex_infection += 1
            elif 0 < infected_neighbors_count and random.uniform(0, 1) <= prob_complex:
                new_infection_marker = True
                new_infected_nodes.append(node)
                n_simple_infection += 1

    infected_list.extend(new_infected_nodes)
    return new_infection_marker, infected_list, n_simple_infection, n_complex_infection

# --- Simple Contagion Process ---
def perform_simple_contagion_unweighted(beta_simple, graph, initially_infected, stop_ratio):
    """
    Simulate the simple contagion process until stop_ratio of nodes are infected.
    """
    infected_list = initially_infected[:]
    total_nodes = len(graph.nodes())
    target_infected = int(stop_ratio * total_nodes)
    time_current = 0
    step_of_infection_per_node = [None] * len(graph.nodes)

    for node_index in initially_infected:
        node_pos = list(graph.nodes).index(node_index)
        step_of_infection_per_node[node_pos] = time_current

    while len(infected_list) < target_infected:
        time_current += 1
        new_infections, infected_list, _, _ = perform_contagion_transfers_unweighted_version(
            graph, infected_list, mode='simple', beta_simple=beta_simple)

        for node_index in infected_list:
            node_pos = list(graph.nodes).index(node_index)
            if step_of_infection_per_node[node_pos] is None:
                step_of_infection_per_node[node_pos] = time_current

    return np.array(step_of_infection_per_node)

# --- Complex Contagion Process ---
def perform_complex_contagion_unweighted(threshold, graph, initially_infected, prob_complex, stop_ratio=None):
    """
    Simulate complex contagion with thresholding and optional probabilistic component.
    """
    complex_contagion_count = 0
    simple_contagion_count = 0
    infected_list = initially_infected[:]
    total_nodes = len(graph.nodes())
    target_infected = int(stop_ratio * total_nodes) if stop_ratio and prob_complex != -1 else None
    time_current = 0
    step_of_infection_per_node = [None] * len(graph.nodes)

    for node_index in initially_infected:
        node_pos = list(graph.nodes).index(node_index)
        step_of_infection_per_node[node_pos] = time_current

    if prob_complex == -1:
        while True:
            time_current += 1
            new_infections, infected_list, n_simp, n_comp = perform_contagion_transfers_unweighted_version(
                graph, infected_list, mode='complex', threshold_complex=threshold, prob_complex=prob_complex)
            complex_contagion_count += n_comp
            simple_contagion_count += n_simp
            if not new_infections:
                break
            for node_index in infected_list:
                node_pos = list(graph.nodes).index(node_index)
                if step_of_infection_per_node[node_pos] is None:
                    step_of_infection_per_node[node_pos] = time_current
    else:
        while len(infected_list) < target_infected:
            time_current += 1
            new_infections, infected_list, n_simp, n_comp = perform_contagion_transfers_unweighted_version(
                graph, infected_list, mode='complex', threshold_complex=threshold, prob_complex=prob_complex)
            complex_contagion_count += n_comp
            simple_contagion_count += n_simp
            for node_index in infected_list:
                node_pos = list(graph.nodes).index(node_index)
                if step_of_infection_per_node[node_pos] is None:
                    step_of_infection_per_node[node_pos] = time_current

    simple_infection_rate = 0
    if complex_contagion_count > 0:
        simple_infection_rate = simple_contagion_count / complex_contagion_count

    return np.array(step_of_infection_per_node), simple_infection_rate

# --- Simulation Runner ---
def run_simulation(network_name, contagion_type, params, iterations, output_file):
    """
    Run multiple contagion simulations for a given network and save results.
    """
    graph_path = os.path.join(NETWORK_DIR, f'G_unweighted_{network_name}.graphml')
    G = load_graph(graph_path)

    order_collection_simulations = []
    seed_sim = []
    contagion_param_sim = []
    simple_ratio_infection_list = []

    for sample_it in tqdm(range(iterations)):
        list_inf = init_nodes_var_reduction(sample_it, list(G.nodes()), params.get("initial_infected", 10))

        if contagion_type == 'simple':
            beta = params['beta']
            stop_ratio = params['stop_ratio']
            order = perform_simple_contagion_unweighted(beta_simple=beta, graph=copy.deepcopy(G),
                                                        initially_infected=copy.deepcopy(list_inf), stop_ratio=stop_ratio)
            contagion_param_sim.append(beta)
            simple_ratio_infection_list.append(1)

        elif contagion_type == 'complex':
            threshold = params['threshold']
            prob_complex = params['prob_complex']
            stop_ratio = params.get('stop_ratio', None)
            order, simple_rate = perform_complex_contagion_unweighted(threshold=threshold, graph=copy.deepcopy(G),
                                                                      initially_infected=copy.deepcopy(list_inf),
                                                                      prob_complex=prob_complex, stop_ratio=stop_ratio)
            contagion_param_sim.append(threshold)
            simple_ratio_infection_list.append(simple_rate)

        order_collection_simulations.append(order)
        seed_sim.append(sample_it)

    result = pd.DataFrame(order_collection_simulations)
    result['seed'] = seed_sim
    result[contagion_type + '_param'] = contagion_param_sim
    result['simple_infection_ratio'] = simple_ratio_infection_list

    result.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

# --- Helper for Node Initialization ---
def init_nodes_var_reduction(iter_cnt, arr, k):
    """
    Initialize k random seed nodes using reproducible RNG.
    """
    rng = np.random.RandomState(iter_cnt)
    return list(rng.choice(arr, k, replace=False))

# --- Main Simulation Loop ---
def main():
    # Choose your networks here
    network_list = [
 'smallworld_p0.1_seed0',
 'smallworld_p0.1_seed1',
 'smallworld_p0.1_seed2',
 'smallworld_p0.1_seed3',
 'smallworld_p0.1_seed4',
 'smallworld_p0.1_seed5',
 'smallworld_p0.1_seed6',
 'smallworld_p0.1_seed7',
 'smallworld_p0.1_seed8',
 'smallworld_p0.1_seed9',
 'smallworld_p0.1_seed10',
 'smallworld_p0.1_seed11',
 'smallworld_p0.1_seed12',
 'smallworld_p0.1_seed13',
 'smallworld_p0.1_seed14',
 'smallworld_p0.1_seed15',
 'smallworld_p0.1_seed16',
 'smallworld_p0.1_seed17',
 'smallworld_p0.1_seed18',
 'smallworld_p0.1_seed19',
 'smallworld_p0.1_seed20',
 'smallworld_p0.1_seed21',
 'smallworld_p0.1_seed22',
 'smallworld_p0.1_seed23',
 'smallworld_p0.1_seed24',
 'smallworld_p0.1_seed25',
 'smallworld_p0.1_seed26',
 'smallworld_p0.1_seed27',
 'smallworld_p0.1_seed28',
 'smallworld_p0.1_seed29',
 'smallworld_p0.1_seed30',
 'smallworld_p0.1_seed31',
 'smallworld_p0.1_seed32',
 'smallworld_p0.1_seed33',
 'smallworld_p0.1_seed34',
 'smallworld_p0.1_seed35',
 'smallworld_p0.1_seed36',
 'smallworld_p0.1_seed37',
 'smallworld_p0.1_seed38',
 'smallworld_p0.1_seed39',
 'smallworld_p0.1_seed40',
 'smallworld_p0.1_seed41',
 'smallworld_p0.1_seed42',
 'smallworld_p0.1_seed43',
 'smallworld_p0.1_seed44',
 'smallworld_p0.1_seed45',
 'smallworld_p0.1_seed46',
 'smallworld_p0.1_seed47',
 'smallworld_p0.1_seed48',
 'smallworld_p0.1_seed49',
 'smallworld_p0.1_seed50',
 'smallworld_p0.1_seed51',
 'smallworld_p0.1_seed52',
 'smallworld_p0.1_seed53',
 'smallworld_p0.1_seed54',
 'smallworld_p0.1_seed55',
 'smallworld_p0.1_seed56',
 'smallworld_p0.1_seed57',
 'smallworld_p0.1_seed58',
 'smallworld_p0.1_seed59',
 'smallworld_p0.1_seed60',
 'smallworld_p0.1_seed61',
 'smallworld_p0.1_seed62',
 'smallworld_p0.1_seed63',
 'smallworld_p0.1_seed64',
 'smallworld_p0.1_seed65',
 'smallworld_p0.1_seed66',
 'smallworld_p0.1_seed67',
 'smallworld_p0.1_seed68',
 'smallworld_p0.1_seed69',
 'smallworld_p0.1_seed70',
 'smallworld_p0.1_seed71',
 'smallworld_p0.1_seed72',
 'smallworld_p0.1_seed73',
 'smallworld_p0.1_seed74',
 'smallworld_p0.1_seed75',
 'smallworld_p0.1_seed76',
 'smallworld_p0.1_seed77',
 'smallworld_p0.1_seed78',
 'smallworld_p0.1_seed79',
 'smallworld_p0.1_seed80',
 'smallworld_p0.1_seed81',
 'smallworld_p0.1_seed82',
 'smallworld_p0.1_seed83',
 'smallworld_p0.1_seed84',
 'smallworld_p0.1_seed85',
 'smallworld_p0.1_seed86',
 'smallworld_p0.1_seed87',
 'smallworld_p0.1_seed88',
 'smallworld_p0.1_seed89',
 'smallworld_p0.1_seed90',
 'smallworld_p0.1_seed91',
 'smallworld_p0.1_seed92',
 'smallworld_p0.1_seed93',
 'smallworld_p0.1_seed94',
 'smallworld_p0.1_seed95',
 'smallworld_p0.1_seed96',
 'smallworld_p0.1_seed97',
 'smallworld_p0.1_seed98',
 'smallworld_p0.1_seed99',
 'smallworld_p0.2_seed0',
 'smallworld_p0.2_seed1',
 'smallworld_p0.2_seed2',
 'smallworld_p0.2_seed3',
 'smallworld_p0.2_seed4',
 'smallworld_p0.2_seed5',
 'smallworld_p0.2_seed6',
 'smallworld_p0.2_seed7',
 'smallworld_p0.2_seed8',
 'smallworld_p0.2_seed9',
 'smallworld_p0.2_seed10',
 'smallworld_p0.2_seed11',
 'smallworld_p0.2_seed12',
 'smallworld_p0.2_seed13',
 'smallworld_p0.2_seed14',
 'smallworld_p0.2_seed15',
 'smallworld_p0.2_seed16',
 'smallworld_p0.2_seed17',
 'smallworld_p0.2_seed18',
 'smallworld_p0.2_seed19',
 'smallworld_p0.2_seed20',
 'smallworld_p0.2_seed21',
 'smallworld_p0.2_seed22',
 'smallworld_p0.2_seed23',
 'smallworld_p0.2_seed24',
 'smallworld_p0.2_seed25',
 'smallworld_p0.2_seed26',
 'smallworld_p0.2_seed27',
 'smallworld_p0.2_seed28',
 'smallworld_p0.2_seed29',
 'smallworld_p0.2_seed30',
 'smallworld_p0.2_seed31',
 'smallworld_p0.2_seed32',
 'smallworld_p0.2_seed33',
 'smallworld_p0.2_seed34',
 'smallworld_p0.2_seed35',
 'smallworld_p0.2_seed36',
 'smallworld_p0.2_seed37',
 'smallworld_p0.2_seed38',
 'smallworld_p0.2_seed39',
 'smallworld_p0.2_seed40',
 'smallworld_p0.2_seed41',
 'smallworld_p0.2_seed42',
 'smallworld_p0.2_seed43',
 'smallworld_p0.2_seed44',
 'smallworld_p0.2_seed45',
 'smallworld_p0.2_seed46',
 'smallworld_p0.2_seed47',
 'smallworld_p0.2_seed48',
 'smallworld_p0.2_seed49',
 'smallworld_p0.2_seed50',
 'smallworld_p0.2_seed51',
 'smallworld_p0.2_seed52',
 'smallworld_p0.2_seed53',
 'smallworld_p0.2_seed54',
 'smallworld_p0.2_seed55',
 'smallworld_p0.2_seed56',
 'smallworld_p0.2_seed57',
 'smallworld_p0.2_seed58',
 'smallworld_p0.2_seed59',
 'smallworld_p0.2_seed60',
 'smallworld_p0.2_seed61',
 'smallworld_p0.2_seed62',
 'smallworld_p0.2_seed63',
 'smallworld_p0.2_seed64',
 'smallworld_p0.2_seed65',
 'smallworld_p0.2_seed66',
 'smallworld_p0.2_seed67',
 'smallworld_p0.2_seed68',
 'smallworld_p0.2_seed69',
 'smallworld_p0.2_seed70',
 'smallworld_p0.2_seed71',
 'smallworld_p0.2_seed72',
 'smallworld_p0.2_seed73',
 'smallworld_p0.2_seed74',
 'smallworld_p0.2_seed75',
 'smallworld_p0.2_seed76',
 'smallworld_p0.2_seed77',
 'smallworld_p0.2_seed78',
 'smallworld_p0.2_seed79',
 'smallworld_p0.2_seed80',
 'smallworld_p0.2_seed81',
 'smallworld_p0.2_seed82',
 'smallworld_p0.2_seed83',
 'smallworld_p0.2_seed84',
 'smallworld_p0.2_seed85',
 'smallworld_p0.2_seed86',
 'smallworld_p0.2_seed87',
 'smallworld_p0.2_seed88',
 'smallworld_p0.2_seed89',
 'smallworld_p0.2_seed90',
 'smallworld_p0.2_seed91',
 'smallworld_p0.2_seed92',
 'smallworld_p0.2_seed93',
 'smallworld_p0.2_seed94',
 'smallworld_p0.2_seed95',
 'smallworld_p0.2_seed96',
 'smallworld_p0.2_seed97',
 'smallworld_p0.2_seed98',
 'smallworld_p0.2_seed99',
 'smallworld_p0.3_seed0',
 'smallworld_p0.3_seed1',
 'smallworld_p0.3_seed2',
 'smallworld_p0.3_seed3',
 'smallworld_p0.3_seed4',
 'smallworld_p0.3_seed5',
 'smallworld_p0.3_seed6',
 'smallworld_p0.3_seed7',
 'smallworld_p0.3_seed8',
 'smallworld_p0.3_seed9',
 'smallworld_p0.3_seed10',
 'smallworld_p0.3_seed11',
 'smallworld_p0.3_seed12',
 'smallworld_p0.3_seed13',
 'smallworld_p0.3_seed14',
 'smallworld_p0.3_seed15',
 'smallworld_p0.3_seed16',
 'smallworld_p0.3_seed17',
 'smallworld_p0.3_seed18',
 'smallworld_p0.3_seed19',
 'smallworld_p0.3_seed20',
 'smallworld_p0.3_seed21',
 'smallworld_p0.3_seed22',
 'smallworld_p0.3_seed23',
 'smallworld_p0.3_seed24',
 'smallworld_p0.3_seed25',
 'smallworld_p0.3_seed26',
 'smallworld_p0.3_seed27',
 'smallworld_p0.3_seed28',
 'smallworld_p0.3_seed29',
 'smallworld_p0.3_seed30',
 'smallworld_p0.3_seed31',
 'smallworld_p0.3_seed32',
 'smallworld_p0.3_seed33',
 'smallworld_p0.3_seed34',
 'smallworld_p0.3_seed35',
 'smallworld_p0.3_seed36',
 'smallworld_p0.3_seed37',
 'smallworld_p0.3_seed38',
 'smallworld_p0.3_seed39',
 'smallworld_p0.3_seed40',
 'smallworld_p0.3_seed41',
 'smallworld_p0.3_seed42',
 'smallworld_p0.3_seed43',
 'smallworld_p0.3_seed44',
 'smallworld_p0.3_seed45',
 'smallworld_p0.3_seed46',
 'smallworld_p0.3_seed47',
 'smallworld_p0.3_seed48',
 'smallworld_p0.3_seed49',
 'smallworld_p0.3_seed50',
 'smallworld_p0.3_seed51',
 'smallworld_p0.3_seed52',
 'smallworld_p0.3_seed53',
 'smallworld_p0.3_seed54',
 'smallworld_p0.3_seed55',
 'smallworld_p0.3_seed56',
 'smallworld_p0.3_seed57',
 'smallworld_p0.3_seed58',
 'smallworld_p0.3_seed59',
 'smallworld_p0.3_seed60',
 'smallworld_p0.3_seed61',
 'smallworld_p0.3_seed62',
 'smallworld_p0.3_seed63',
 'smallworld_p0.3_seed64',
 'smallworld_p0.3_seed65',
 'smallworld_p0.3_seed66',
 'smallworld_p0.3_seed67',
 'smallworld_p0.3_seed68',
 'smallworld_p0.3_seed69',
 'smallworld_p0.3_seed70',
 'smallworld_p0.3_seed71',
 'smallworld_p0.3_seed72',
 'smallworld_p0.3_seed73',
 'smallworld_p0.3_seed74',
 'smallworld_p0.3_seed75',
 'smallworld_p0.3_seed76',
 'smallworld_p0.3_seed77',
 'smallworld_p0.3_seed78',
 'smallworld_p0.3_seed79',
 'smallworld_p0.3_seed80',
 'smallworld_p0.3_seed81',
 'smallworld_p0.3_seed82',
 'smallworld_p0.3_seed83',
 'smallworld_p0.3_seed84',
 'smallworld_p0.3_seed85',
 'smallworld_p0.3_seed86',
 'smallworld_p0.3_seed87',
 'smallworld_p0.3_seed88',
 'smallworld_p0.3_seed89',
 'smallworld_p0.3_seed90',
 'smallworld_p0.3_seed91',
 'smallworld_p0.3_seed92',
 'smallworld_p0.3_seed93',
 'smallworld_p0.3_seed94',
 'smallworld_p0.3_seed95',
 'smallworld_p0.3_seed96',
 'smallworld_p0.3_seed97',
 'smallworld_p0.3_seed98',
 'smallworld_p0.3_seed99',
 'smallworld_p0.4_seed0',
 'smallworld_p0.4_seed1',
 'smallworld_p0.4_seed2',
 'smallworld_p0.4_seed3',
 'smallworld_p0.4_seed4',
 'smallworld_p0.4_seed5',
 'smallworld_p0.4_seed6',
 'smallworld_p0.4_seed7',
 'smallworld_p0.4_seed8',
 'smallworld_p0.4_seed9',
 'smallworld_p0.4_seed10',
 'smallworld_p0.4_seed11',
 'smallworld_p0.4_seed12',
 'smallworld_p0.4_seed13',
 'smallworld_p0.4_seed14',
 'smallworld_p0.4_seed15',
 'smallworld_p0.4_seed16',
 'smallworld_p0.4_seed17',
 'smallworld_p0.4_seed18',
 'smallworld_p0.4_seed19',
 'smallworld_p0.4_seed20',
 'smallworld_p0.4_seed21',
 'smallworld_p0.4_seed22',
 'smallworld_p0.4_seed23',
 'smallworld_p0.4_seed24',
 'smallworld_p0.4_seed25',
 'smallworld_p0.4_seed26',
 'smallworld_p0.4_seed27',
 'smallworld_p0.4_seed28',
 'smallworld_p0.4_seed29',
 'smallworld_p0.4_seed30',
 'smallworld_p0.4_seed31',
 'smallworld_p0.4_seed32',
 'smallworld_p0.4_seed33',
 'smallworld_p0.4_seed34',
 'smallworld_p0.4_seed35',
 'smallworld_p0.4_seed36',
 'smallworld_p0.4_seed37',
 'smallworld_p0.4_seed38',
 'smallworld_p0.4_seed39',
 'smallworld_p0.4_seed40',
 'smallworld_p0.4_seed41',
 'smallworld_p0.4_seed42',
 'smallworld_p0.4_seed43',
 'smallworld_p0.4_seed44',
 'smallworld_p0.4_seed45',
 'smallworld_p0.4_seed46',
 'smallworld_p0.4_seed47',
 'smallworld_p0.4_seed48',
 'smallworld_p0.4_seed49',
 'smallworld_p0.4_seed50',
 'smallworld_p0.4_seed51',
 'smallworld_p0.4_seed52',
 'smallworld_p0.4_seed53',
 'smallworld_p0.4_seed54',
 'smallworld_p0.4_seed55',
 'smallworld_p0.4_seed56',
 'smallworld_p0.4_seed57',
 'smallworld_p0.4_seed58',
 'smallworld_p0.4_seed59',
 'smallworld_p0.4_seed60',
 'smallworld_p0.4_seed61',
 'smallworld_p0.4_seed62',
 'smallworld_p0.4_seed63',
 'smallworld_p0.4_seed64',
 'smallworld_p0.4_seed65',
 'smallworld_p0.4_seed66',
 'smallworld_p0.4_seed67',
 'smallworld_p0.4_seed68',
 'smallworld_p0.4_seed69',
 'smallworld_p0.4_seed70',
 'smallworld_p0.4_seed71',
 'smallworld_p0.4_seed72',
 'smallworld_p0.4_seed73',
 'smallworld_p0.4_seed74',
 'smallworld_p0.4_seed75',
 'smallworld_p0.4_seed76',
 'smallworld_p0.4_seed77',
 'smallworld_p0.4_seed78',
 'smallworld_p0.4_seed79',
 'smallworld_p0.4_seed80',
 'smallworld_p0.4_seed81',
 'smallworld_p0.4_seed82',
 'smallworld_p0.4_seed83',
 'smallworld_p0.4_seed84',
 'smallworld_p0.4_seed85',
 'smallworld_p0.4_seed86',
 'smallworld_p0.4_seed87',
 'smallworld_p0.4_seed88',
 'smallworld_p0.4_seed89',
 'smallworld_p0.4_seed90',
 'smallworld_p0.4_seed91',
 'smallworld_p0.4_seed92',
 'smallworld_p0.4_seed93',
 'smallworld_p0.4_seed94',
 'smallworld_p0.4_seed95',
 'smallworld_p0.4_seed96',
 'smallworld_p0.4_seed97',
 'smallworld_p0.4_seed98',
 'smallworld_p0.4_seed99',
 'smallworld_p0.5_seed0',
 'smallworld_p0.5_seed1',
 'smallworld_p0.5_seed2',
 'smallworld_p0.5_seed3',
 'smallworld_p0.5_seed4',
 'smallworld_p0.5_seed5',
 'smallworld_p0.5_seed6',
 'smallworld_p0.5_seed7',
 'smallworld_p0.5_seed8',
 'smallworld_p0.5_seed9',
 'smallworld_p0.5_seed10',
 'smallworld_p0.5_seed11',
 'smallworld_p0.5_seed12',
 'smallworld_p0.5_seed13',
 'smallworld_p0.5_seed14',
 'smallworld_p0.5_seed15',
 'smallworld_p0.5_seed16',
 'smallworld_p0.5_seed17',
 'smallworld_p0.5_seed18',
 'smallworld_p0.5_seed19',
 'smallworld_p0.5_seed20',
 'smallworld_p0.5_seed21',
 'smallworld_p0.5_seed22',
 'smallworld_p0.5_seed23',
 'smallworld_p0.5_seed24',
 'smallworld_p0.5_seed25',
 'smallworld_p0.5_seed26',
 'smallworld_p0.5_seed27',
 'smallworld_p0.5_seed28',
 'smallworld_p0.5_seed29',
 'smallworld_p0.5_seed30',
 'smallworld_p0.5_seed31',
 'smallworld_p0.5_seed32',
 'smallworld_p0.5_seed33',
 'smallworld_p0.5_seed34',
 'smallworld_p0.5_seed35',
 'smallworld_p0.5_seed36',
 'smallworld_p0.5_seed37',
 'smallworld_p0.5_seed38',
 'smallworld_p0.5_seed39',
 'smallworld_p0.5_seed40',
 'smallworld_p0.5_seed41',
 'smallworld_p0.5_seed42',
 'smallworld_p0.5_seed43',
 'smallworld_p0.5_seed44',
 'smallworld_p0.5_seed45',
 'smallworld_p0.5_seed46',
 'smallworld_p0.5_seed47',
 'smallworld_p0.5_seed48',
 'smallworld_p0.5_seed49',
 'smallworld_p0.5_seed50',
 'smallworld_p0.5_seed51',
 'smallworld_p0.5_seed52',
 'smallworld_p0.5_seed53',
 'smallworld_p0.5_seed54',
 'smallworld_p0.5_seed55',
 'smallworld_p0.5_seed56',
 'smallworld_p0.5_seed57',
 'smallworld_p0.5_seed58',
 'smallworld_p0.5_seed59',
 'smallworld_p0.5_seed60',
 'smallworld_p0.5_seed61',
 'smallworld_p0.5_seed62',
 'smallworld_p0.5_seed63',
 'smallworld_p0.5_seed64',
 'smallworld_p0.5_seed65',
 'smallworld_p0.5_seed66',
 'smallworld_p0.5_seed67',
 'smallworld_p0.5_seed68',
 'smallworld_p0.5_seed69',
 'smallworld_p0.5_seed70',
 'smallworld_p0.5_seed71',
 'smallworld_p0.5_seed72',
 'smallworld_p0.5_seed73',
 'smallworld_p0.5_seed74',
 'smallworld_p0.5_seed75',
 'smallworld_p0.5_seed76',
 'smallworld_p0.5_seed77',
 'smallworld_p0.5_seed78',
 'smallworld_p0.5_seed79',
 'smallworld_p0.5_seed80',
 'smallworld_p0.5_seed81',
 'smallworld_p0.5_seed82',
 'smallworld_p0.5_seed83',
 'smallworld_p0.5_seed84',
 'smallworld_p0.5_seed85',
 'smallworld_p0.5_seed86',
 'smallworld_p0.5_seed87',
 'smallworld_p0.5_seed88',
 'smallworld_p0.5_seed89',
 'smallworld_p0.5_seed90',
 'smallworld_p0.5_seed91',
 'smallworld_p0.5_seed92',
 'smallworld_p0.5_seed93',
 'smallworld_p0.5_seed94',
 'smallworld_p0.5_seed95',
 'smallworld_p0.5_seed96',
 'smallworld_p0.5_seed97',
 'smallworld_p0.5_seed98',
 'smallworld_p0.5_seed99',
 'smallworld_p0.6_seed0',
 'smallworld_p0.6_seed1',
 'smallworld_p0.6_seed2',
 'smallworld_p0.6_seed3',
 'smallworld_p0.6_seed4',
 'smallworld_p0.6_seed5',
 'smallworld_p0.6_seed6',
 'smallworld_p0.6_seed7',
 'smallworld_p0.6_seed8',
 'smallworld_p0.6_seed9',
 'smallworld_p0.6_seed10',
 'smallworld_p0.6_seed11',
 'smallworld_p0.6_seed12',
 'smallworld_p0.6_seed13',
 'smallworld_p0.6_seed14',
 'smallworld_p0.6_seed15',
 'smallworld_p0.6_seed16',
 'smallworld_p0.6_seed17',
 'smallworld_p0.6_seed18',
 'smallworld_p0.6_seed19',
 'smallworld_p0.6_seed20',
 'smallworld_p0.6_seed21',
 'smallworld_p0.6_seed22',
 'smallworld_p0.6_seed23',
 'smallworld_p0.6_seed24',
 'smallworld_p0.6_seed25',
 'smallworld_p0.6_seed26',
 'smallworld_p0.6_seed27',
 'smallworld_p0.6_seed28',
 'smallworld_p0.6_seed29',
 'smallworld_p0.6_seed30',
 'smallworld_p0.6_seed31',
 'smallworld_p0.6_seed32',
 'smallworld_p0.6_seed33',
 'smallworld_p0.6_seed34',
 'smallworld_p0.6_seed35',
 'smallworld_p0.6_seed36',
 'smallworld_p0.6_seed37',
 'smallworld_p0.6_seed38',
 'smallworld_p0.6_seed39',
 'smallworld_p0.6_seed40',
 'smallworld_p0.6_seed41',
 'smallworld_p0.6_seed42',
 'smallworld_p0.6_seed43',
 'smallworld_p0.6_seed44',
 'smallworld_p0.6_seed45',
 'smallworld_p0.6_seed46',
 'smallworld_p0.6_seed47',
 'smallworld_p0.6_seed48',
 'smallworld_p0.6_seed49',
 'smallworld_p0.6_seed50',
 'smallworld_p0.6_seed51',
 'smallworld_p0.6_seed52',
 'smallworld_p0.6_seed53',
 'smallworld_p0.6_seed54',
 'smallworld_p0.6_seed55',
 'smallworld_p0.6_seed56',
 'smallworld_p0.6_seed57',
 'smallworld_p0.6_seed58',
 'smallworld_p0.6_seed59',
 'smallworld_p0.6_seed60',
 'smallworld_p0.6_seed61',
 'smallworld_p0.6_seed62',
 'smallworld_p0.6_seed63',
 'smallworld_p0.6_seed64',
 'smallworld_p0.6_seed65',
 'smallworld_p0.6_seed66',
 'smallworld_p0.6_seed67',
 'smallworld_p0.6_seed68',
 'smallworld_p0.6_seed69',
 'smallworld_p0.6_seed70',
 'smallworld_p0.6_seed71',
 'smallworld_p0.6_seed72',
 'smallworld_p0.6_seed73',
 'smallworld_p0.6_seed74',
 'smallworld_p0.6_seed75',
 'smallworld_p0.6_seed76',
 'smallworld_p0.6_seed77',
 'smallworld_p0.6_seed78',
 'smallworld_p0.6_seed79',
 'smallworld_p0.6_seed80',
 'smallworld_p0.6_seed81',
 'smallworld_p0.6_seed82',
 'smallworld_p0.6_seed83',
 'smallworld_p0.6_seed84',
 'smallworld_p0.6_seed85',
 'smallworld_p0.6_seed86',
 'smallworld_p0.6_seed87',
 'smallworld_p0.6_seed88',
 'smallworld_p0.6_seed89',
 'smallworld_p0.6_seed90',
 'smallworld_p0.6_seed91',
 'smallworld_p0.6_seed92',
 'smallworld_p0.6_seed93',
 'smallworld_p0.6_seed94',
 'smallworld_p0.6_seed95',
 'smallworld_p0.6_seed96',
 'smallworld_p0.6_seed97',
 'smallworld_p0.6_seed98',
 'smallworld_p0.6_seed99',
 'smallworld_p0.7_seed0',
 'smallworld_p0.7_seed1',
 'smallworld_p0.7_seed2',
 'smallworld_p0.7_seed3',
 'smallworld_p0.7_seed4',
 'smallworld_p0.7_seed5',
 'smallworld_p0.7_seed6',
 'smallworld_p0.7_seed7',
 'smallworld_p0.7_seed8',
 'smallworld_p0.7_seed9',
 'smallworld_p0.7_seed10',
 'smallworld_p0.7_seed11',
 'smallworld_p0.7_seed12',
 'smallworld_p0.7_seed13',
 'smallworld_p0.7_seed14',
 'smallworld_p0.7_seed15',
 'smallworld_p0.7_seed16',
 'smallworld_p0.7_seed17',
 'smallworld_p0.7_seed18',
 'smallworld_p0.7_seed19',
 'smallworld_p0.7_seed20',
 'smallworld_p0.7_seed21',
 'smallworld_p0.7_seed22',
 'smallworld_p0.7_seed23',
 'smallworld_p0.7_seed24',
 'smallworld_p0.7_seed25',
 'smallworld_p0.7_seed26',
 'smallworld_p0.7_seed27',
 'smallworld_p0.7_seed28',
 'smallworld_p0.7_seed29',
 'smallworld_p0.7_seed30',
 'smallworld_p0.7_seed31',
 'smallworld_p0.7_seed32',
 'smallworld_p0.7_seed33',
 'smallworld_p0.7_seed34',
 'smallworld_p0.7_seed35',
 'smallworld_p0.7_seed36',
 'smallworld_p0.7_seed37',
 'smallworld_p0.7_seed38',
 'smallworld_p0.7_seed39',
 'smallworld_p0.7_seed40',
 'smallworld_p0.7_seed41',
 'smallworld_p0.7_seed42',
 'smallworld_p0.7_seed43',
 'smallworld_p0.7_seed44',
 'smallworld_p0.7_seed45',
 'smallworld_p0.7_seed46',
 'smallworld_p0.7_seed47',
 'smallworld_p0.7_seed48',
 'smallworld_p0.7_seed49',
 'smallworld_p0.7_seed50',
 'smallworld_p0.7_seed51',
 'smallworld_p0.7_seed52',
 'smallworld_p0.7_seed53',
 'smallworld_p0.7_seed54',
 'smallworld_p0.7_seed55',
 'smallworld_p0.7_seed56',
 'smallworld_p0.7_seed57',
 'smallworld_p0.7_seed58',
 'smallworld_p0.7_seed59',
 'smallworld_p0.7_seed60',
 'smallworld_p0.7_seed61',
 'smallworld_p0.7_seed62',
 'smallworld_p0.7_seed63',
 'smallworld_p0.7_seed64',
 'smallworld_p0.7_seed65',
 'smallworld_p0.7_seed66',
 'smallworld_p0.7_seed67',
 'smallworld_p0.7_seed68',
 'smallworld_p0.7_seed69',
 'smallworld_p0.7_seed70',
 'smallworld_p0.7_seed71',
 'smallworld_p0.7_seed72',
 'smallworld_p0.7_seed73',
 'smallworld_p0.7_seed74',
 'smallworld_p0.7_seed75',
 'smallworld_p0.7_seed76',
 'smallworld_p0.7_seed77',
 'smallworld_p0.7_seed78',
 'smallworld_p0.7_seed79',
 'smallworld_p0.7_seed80',
 'smallworld_p0.7_seed81',
 'smallworld_p0.7_seed82',
 'smallworld_p0.7_seed83',
 'smallworld_p0.7_seed84',
 'smallworld_p0.7_seed85',
 'smallworld_p0.7_seed86',
 'smallworld_p0.7_seed87',
 'smallworld_p0.7_seed88',
 'smallworld_p0.7_seed89',
 'smallworld_p0.7_seed90',
 'smallworld_p0.7_seed91',
 'smallworld_p0.7_seed92',
 'smallworld_p0.7_seed93',
 'smallworld_p0.7_seed94',
 'smallworld_p0.7_seed95',
 'smallworld_p0.7_seed96',
 'smallworld_p0.7_seed97',
 'smallworld_p0.7_seed98',
 'smallworld_p0.7_seed99',
 'smallworld_p0.8_seed0',
 'smallworld_p0.8_seed1',
 'smallworld_p0.8_seed2',
 'smallworld_p0.8_seed3',
 'smallworld_p0.8_seed4',
 'smallworld_p0.8_seed5',
 'smallworld_p0.8_seed6',
 'smallworld_p0.8_seed7',
 'smallworld_p0.8_seed8',
 'smallworld_p0.8_seed9',
 'smallworld_p0.8_seed10',
 'smallworld_p0.8_seed11',
 'smallworld_p0.8_seed12',
 'smallworld_p0.8_seed13',
 'smallworld_p0.8_seed14',
 'smallworld_p0.8_seed15',
 'smallworld_p0.8_seed16',
 'smallworld_p0.8_seed17',
 'smallworld_p0.8_seed18',
 'smallworld_p0.8_seed19',
 'smallworld_p0.8_seed20',
 'smallworld_p0.8_seed21',
 'smallworld_p0.8_seed22',
 'smallworld_p0.8_seed23',
 'smallworld_p0.8_seed24',
 'smallworld_p0.8_seed25',
 'smallworld_p0.8_seed26',
 'smallworld_p0.8_seed27',
 'smallworld_p0.8_seed28',
 'smallworld_p0.8_seed29',
 'smallworld_p0.8_seed30',
 'smallworld_p0.8_seed31',
 'smallworld_p0.8_seed32',
 'smallworld_p0.8_seed33',
 'smallworld_p0.8_seed34',
 'smallworld_p0.8_seed35',
 'smallworld_p0.8_seed36',
 'smallworld_p0.8_seed37',
 'smallworld_p0.8_seed38',
 'smallworld_p0.8_seed39',
 'smallworld_p0.8_seed40',
 'smallworld_p0.8_seed41',
 'smallworld_p0.8_seed42',
 'smallworld_p0.8_seed43',
 'smallworld_p0.8_seed44',
 'smallworld_p0.8_seed45',
 'smallworld_p0.8_seed46',
 'smallworld_p0.8_seed47',
 'smallworld_p0.8_seed48',
 'smallworld_p0.8_seed49',
 'smallworld_p0.8_seed50',
 'smallworld_p0.8_seed51',
 'smallworld_p0.8_seed52',
 'smallworld_p0.8_seed53',
 'smallworld_p0.8_seed54',
 'smallworld_p0.8_seed55',
 'smallworld_p0.8_seed56',
 'smallworld_p0.8_seed57',
 'smallworld_p0.8_seed58',
 'smallworld_p0.8_seed59',
 'smallworld_p0.8_seed60',
 'smallworld_p0.8_seed61',
 'smallworld_p0.8_seed62',
 'smallworld_p0.8_seed63',
 'smallworld_p0.8_seed64',
 'smallworld_p0.8_seed65',
 'smallworld_p0.8_seed66',
 'smallworld_p0.8_seed67',
 'smallworld_p0.8_seed68',
 'smallworld_p0.8_seed69',
 'smallworld_p0.8_seed70',
 'smallworld_p0.8_seed71',
 'smallworld_p0.8_seed72',
 'smallworld_p0.8_seed73',
 'smallworld_p0.8_seed74',
 'smallworld_p0.8_seed75',
 'smallworld_p0.8_seed76',
 'smallworld_p0.8_seed77',
 'smallworld_p0.8_seed78',
 'smallworld_p0.8_seed79',
 'smallworld_p0.8_seed80',
 'smallworld_p0.8_seed81',
 'smallworld_p0.8_seed82',
 'smallworld_p0.8_seed83',
 'smallworld_p0.8_seed84',
 'smallworld_p0.8_seed85',
 'smallworld_p0.8_seed86',
 'smallworld_p0.8_seed87',
 'smallworld_p0.8_seed88',
 'smallworld_p0.8_seed89',
 'smallworld_p0.8_seed90',
 'smallworld_p0.8_seed91',
 'smallworld_p0.8_seed92',
 'smallworld_p0.8_seed93',
 'smallworld_p0.8_seed94',
 'smallworld_p0.8_seed95',
 'smallworld_p0.8_seed96',
 'smallworld_p0.8_seed97',
 'smallworld_p0.8_seed98',
 'smallworld_p0.8_seed99',
 'smallworld_p0.9_seed0',
 'smallworld_p0.9_seed1',
 'smallworld_p0.9_seed2',
 'smallworld_p0.9_seed3',
 'smallworld_p0.9_seed4',
 'smallworld_p0.9_seed5',
 'smallworld_p0.9_seed6',
 'smallworld_p0.9_seed7',
 'smallworld_p0.9_seed8',
 'smallworld_p0.9_seed9',
 'smallworld_p0.9_seed10',
 'smallworld_p0.9_seed11',
 'smallworld_p0.9_seed12',
 'smallworld_p0.9_seed13',
 'smallworld_p0.9_seed14',
 'smallworld_p0.9_seed15',
 'smallworld_p0.9_seed16',
 'smallworld_p0.9_seed17',
 'smallworld_p0.9_seed18',
 'smallworld_p0.9_seed19',
 'smallworld_p0.9_seed20',
 'smallworld_p0.9_seed21',
 'smallworld_p0.9_seed22',
 'smallworld_p0.9_seed23',
 'smallworld_p0.9_seed24',
 'smallworld_p0.9_seed25',
 'smallworld_p0.9_seed26',
 'smallworld_p0.9_seed27',
 'smallworld_p0.9_seed28',
 'smallworld_p0.9_seed29',
 'smallworld_p0.9_seed30',
 'smallworld_p0.9_seed31',
 'smallworld_p0.9_seed32',
 'smallworld_p0.9_seed33',
 'smallworld_p0.9_seed34',
 'smallworld_p0.9_seed35',
 'smallworld_p0.9_seed36',
 'smallworld_p0.9_seed37',
 'smallworld_p0.9_seed38',
 'smallworld_p0.9_seed39',
 'smallworld_p0.9_seed40',
 'smallworld_p0.9_seed41',
 'smallworld_p0.9_seed42',
 'smallworld_p0.9_seed43',
 'smallworld_p0.9_seed44',
 'smallworld_p0.9_seed45',
 'smallworld_p0.9_seed46',
 'smallworld_p0.9_seed47',
 'smallworld_p0.9_seed48',
 'smallworld_p0.9_seed49',
 'smallworld_p0.9_seed50',
 'smallworld_p0.9_seed51',
 'smallworld_p0.9_seed52',
 'smallworld_p0.9_seed53',
 'smallworld_p0.9_seed54',
 'smallworld_p0.9_seed55',
 'smallworld_p0.9_seed56',
 'smallworld_p0.9_seed57',
 'smallworld_p0.9_seed58',
 'smallworld_p0.9_seed59',
 'smallworld_p0.9_seed60',
 'smallworld_p0.9_seed61',
 'smallworld_p0.9_seed62',
 'smallworld_p0.9_seed63',
 'smallworld_p0.9_seed64',
 'smallworld_p0.9_seed65',
 'smallworld_p0.9_seed66',
 'smallworld_p0.9_seed67',
 'smallworld_p0.9_seed68',
 'smallworld_p0.9_seed69',
 'smallworld_p0.9_seed70',
 'smallworld_p0.9_seed71',
 'smallworld_p0.9_seed72',
 'smallworld_p0.9_seed73',
 'smallworld_p0.9_seed74',
 'smallworld_p0.9_seed75',
 'smallworld_p0.9_seed76',
 'smallworld_p0.9_seed77',
 'smallworld_p0.9_seed78',
 'smallworld_p0.9_seed79',
 'smallworld_p0.9_seed80',
 'smallworld_p0.9_seed81',
 'smallworld_p0.9_seed82',
 'smallworld_p0.9_seed83',
 'smallworld_p0.9_seed84',
 'smallworld_p0.9_seed85',
 'smallworld_p0.9_seed86',
 'smallworld_p0.9_seed87',
 'smallworld_p0.9_seed88',
 'smallworld_p0.9_seed89',
 'smallworld_p0.9_seed90',
 'smallworld_p0.9_seed91',
 'smallworld_p0.9_seed92',
 'smallworld_p0.9_seed93',
 'smallworld_p0.9_seed94',
 'smallworld_p0.9_seed95',
 'smallworld_p0.9_seed96',
 'smallworld_p0.9_seed97',
 'smallworld_p0.9_seed98',
 'smallworld_p0.9_seed99']

    iterations = 200
    stop_ratio = 0.85

    # Simple contagion parameters
    simple_betas = [0.02, 0.03, 0.04, 0.05]

    # Complex contagion parameters
    complex_thetas = [2, 3, 4, 5, 6, 7, 8, 9]
    complex_probs = [-1, 0.02, 0.03, 0.04]

    # --- Run Simple Contagion Simulations ---
    for network_name in network_list:
        graph_path = os.path.join(NETWORK_DIR, f'G_unweighted_{network_name}.graphml')
        G = load_graph(graph_path)

        for beta in simple_betas:
            output_file = os.path.join(RESULT_DIR, f'unweighted_simple_{network_name}_beta_{beta}.csv')
            if os.path.exists(output_file):
                print(output_file, "already exists, skipping.")
                continue
            simple_params = {
                'beta': beta,
                'stop_ratio': stop_ratio,
                'initial_infected': int(np.round(len(G.nodes()) * 0.01))
            }
            run_simulation(network_name, 'simple', simple_params, iterations, output_file)

    # --- Run Complex Contagion Simulations ---
    for network_name in network_list:
        graph_path = os.path.join(NETWORK_DIR, f'G_unweighted_{network_name}.graphml')
        G = load_graph(graph_path)

        for theta in complex_thetas:
            for prob_complex in complex_probs:
                output_file = os.path.join(RESULT_DIR, f'unweighted_complex_{network_name}_theta_{theta}_prob_{prob_complex}.csv')
                if os.path.exists(output_file):
                    print(output_file, "already exists, skipping.")
                    continue

                complex_params = {
                    'threshold': theta,
                    'prob_complex': prob_complex,
                    'stop_ratio': None if prob_complex == -1 else stop_ratio,
                    'initial_infected': int(np.round(len(G.nodes()) * 0.01))
                }
                run_simulation(network_name, 'complex', complex_params, iterations, output_file)

if __name__ == "__main__":
    main()
