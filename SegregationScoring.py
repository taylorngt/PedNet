import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics import accuracy_score, auc
import pprint
import numpy as np
from collections import OrderedDict
import powerlaw
import itertools as it
from typing import Dict, Set, Tuple
from scipy.optimize import minimize
import copy
from pprint import pprint
from PedigreeDAGAnalysis import generations, parents, aff, unaff, founder_influence, longest_path_length


#################### MODULAR VARIANT SCORING ####################
'''
Scores: measures of variant association likelihoods accounting for graph/pedigree structure as well as genotype and phenotype data,
provided in mode agnostic form

Current List of Scores:
-------------------------

'''
#----------------------------------------------------------------------
# 1. Edge Consistency
#----------------------------------------------------------------------
def edge_consistency(G, gt):
    """
    Fraction of parent→child edges whose genotype transition is Mendelian-
    compatible under the specified inheritance mode.
    gt is a dict {node: 0/1/2}.

    Given working off of genotypes, mode is irrelevant.
    """
    #partental genotype (pg,mg) | (mg,pg) --> possible child genotypes {}
    BOTH_PARENT_ALLOWED_INHERITENCE = {
        (0,0):{0}, (1,0):{0,1}, (1,1):{0,1,2}, (2,1):{1,2}, (2,0):{1}, (2,2):{2}
    }
    SINGLE_PARENT_ALLOWED_INHERITENCE = {
        0:{0,1}, 1:{0,1,2}, 2:{1,2}
    }

    sequenced_samples = gt.keys()

    good=0; total=0
    for child in G.nodes:
        if child not in sequenced_samples:
            continue

        prnts=parents(G,child)
        sequenced_prnts = [p for p in prnts if p in sequenced_samples]


        if len(sequenced_prnts) == 1:
            if gt[child] in SINGLE_PARENT_ALLOWED_INHERITENCE.get(gt.get(sequenced_prnts[0]), {}):
                good+=1
            total+=1
        elif len(sequenced_prnts) == 2:
            gp,gm=[gt[p] for p in sequenced_prnts]
            par_gt = (gp,gm) if (gp,gm) in BOTH_PARENT_ALLOWED_INHERITENCE.keys() else (gm,gp)
            if gt[child] in BOTH_PARENT_ALLOWED_INHERITENCE.get(par_gt, {}):
                good+=2
            total+=2
        # If neither parent is sequenced, we cannot check Mendelian consistency for this child, so we skip.


    return good/total if total > 0 else 0

# ---------------------------------------------------------------------
# 2. Generation Continuity
# ---------------------------------------------------------------------
def generation_continuity(G, gt):
    """
    Return the fraction of generations with carriers (by genotype)
    """
    gen = generations(G)
    gens_total = max(gen.values())+1
    sequenced_samples = gt.keys()
    alt_gens = {gen[n] for n in sequenced_samples if gt[n]>0}
    sequenced_gens = {gen[n] for n in sequenced_samples}
    return len(alt_gens)/len(sequenced_gens) if sequenced_gens else 0



# ---------------------------------------------------------------------
# 3. Betweeness of Carriers in Affected+Carrier Subgraph
# ---------------------------------------------------------------------
'''
Currently defunct based on necessary inclusion of genotype data
which is not included in pedigree graph alone
'''
def carrier_betweenness(G, gt):
    aff_nodes = aff(G)
    unaff_nodes = unaff(G)
    sequenced_samples = gt.keys()
    sequenced_unaff_nodes = list(set(unaff_nodes) & set(sequenced_samples))
    carrier_nodes = [n for n in sequenced_unaff_nodes if gt[n] == 1]
    carrier_aff_subgraph = G.subgraph(aff_nodes+carrier_nodes)
    subgraph_bet = nx.betweenness_centrality(carrier_aff_subgraph, normalized= False)
    complete_bet = nx.betweenness_centrality(G, normalized= False)
    avg_carrier_betweenness = np.mean([subgraph_bet[n] for n in carrier_aff_subgraph.nodes]) if len(carrier_nodes) > 0 else 0
    avg_complete_betweenness = np.mean([complete_bet[n] for n in G.nodes]) if len(G.nodes) > 0 else 0

    adj_carrier_betweenness = avg_carrier_betweenness/avg_complete_betweenness if avg_complete_betweenness else 0

    return adj_carrier_betweenness

# ---------------------------------------------------------------------
# 4. Average Founder Influence (extended scoring)
# ---------------------------------------------------------------------
def avg_founder_influence(G, gt):
    sequenced_samples = gt.keys()
    founders = [n for n in G if G.in_degree(n)==0 and n in sequenced_samples and gt[n]>0]
    if founders:
        fi = founder_influence(G)
        avg_fi = sum(fi[f] for f in founders) / len(founders)
    else:
        avg_fi = 0
    return avg_fi


# ---------------------------------------------------------------------
# 5. Alternate Allelic Depth (extended scoring)
# ---------------------------------------------------------------------
def alt_depth_ratio(G, gt): 
    depth = longest_path_length(G)
    sequenced_samples = gt.keys()
    alt_nodes = [n for n in sequenced_samples if gt[n]>0]
    alt_depth = 0
    founders = [n for n in G if G.in_degree(n)==0 and n in sequenced_samples and gt[n]>0]
    if depth and founders and alt_nodes:
        # shortest founder→alt path for each pair that is connected
        lengths = []
        for f in founders:
            for a in alt_nodes:
                if nx.has_path(G, f, a):
                    lengths.append(nx.shortest_path_length(G, f, a))
        if lengths:
            alt_depth = max(lengths)
    return alt_depth / depth



# ---------------------------------------------------------------------
# VARIANT CATEGORICAL SCORING WRAPPER
# ---------------------------------------------------------------------
'''
Mode agnostics raw variant scores
'''
def raw_categorical_scoring(G, gt):
    return {
        'edge_consistency': edge_consistency(G, gt),
        'generation_continuity': generation_continuity(G, gt),
        'carrier_betweenness': carrier_betweenness(G, gt), 
        'founder_influence' : avg_founder_influence(G,gt),
        'alt_depth' : alt_depth_ratio(G,gt)
    }


# ---------------------------------------------------------------------
# SEGREGATION SCORING
# ---------------------------------------------------------------------
def segregation_network_score(G, gt, mode, Scoring_Method= 'Original', categorical_scores=0, weights={'w_edge':0.6,'w_gen':0.2,'w_bet':0.2}, verbose= False):

    #Categorical Score Calculation
    if not categorical_scores:

        categorical_scores = {}

        #edge consistency
        categorical_scores['edge_score']= edge_consistency(G,gt)

        # generation continuity
        categorical_scores['gen_score']= max(0,min(1,generation_continuity(G,gt))) #ensures genscore within [0,1]

        # carrier betweenness
        cb = carrier_betweenness(G, gt) if mode=='AR' else 1-carrier_betweenness(G, gt)
        categorical_scores['bet_score']= max(0,min(1,cb))


        #Extended Categorical Scores
        if Scoring_Method == 'Extended':
            #when incorrect number of weights are given for extended scores, resort to default weights
            if len(weights.keys()) < 5:
                weights = {'w_edge': 0.6, 'w_gen': 0.1, 'w_bet': 0.1, 'w_found': 0.1, 'w_depth': 0.1}



    #Weighted Score Calculation
    score = (weights['w_edge'] * categorical_scores['edge_score']) + (weights['w_gen'] * categorical_scores['gen_score']) + (weights['w_bet'] * categorical_scores['bet_score'])
    if Scoring_Method == 'Extended':
        score += (weights['w_found'] * categorical_scores['found_score']) + (weights['w_depth'] * categorical_scores['depth_score'])

    '''
    Current Scoring Metrics:
        Original:
            edge_score
            gen_score
            bet_score
        Extended:
            found_score
            red_score (CURRENTLY UNUSED)
            depth_score
            width_score (CURRENTLY UNUSED)
            cov_score (CURRENTLY UNUSED)
    '''
    if verbose:
        print(f"Edge Score: {categorical_scores['edge_score']}; Gen Score: {categorical_scores['gen_score']}; Bet Score: {categorical_scores['bet_score']}")
        if Scoring_Method == 'Extended':
            print(f"Found Score: {categorical_scores['found_score']}; Depth Score: {categorical_scores['depth_score']}")
        print(f'Segregation Score: {score}')
    return score, categorical_scores