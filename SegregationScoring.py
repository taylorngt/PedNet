import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.optimize import minimize
from PedigreeDAGAnalysis import generations, parents, aff, unaff, founder_influence, longest_path_length
from PedigreeDataGeneration import PedGraph_VarTable_generator
from IPython.display import display

############### GLOBAL DEFAULT PEDIGREE PARAMETERS ##################
TRIAL_COUNT = 500
MAX_CHILDREN = 5
ALT_FREQ_RANGE = (2, 20)
BACKPROP_LIKELIHOOD_RANGE = (25, 75)
SPOUSE_LIKELIHOOD_RANGE = (25, 75)
AFFECTED_SPOUSE = True

############### GLOBAL DEFAULT VARTABLE PARAMETERS #####################
SEQUENCE_COVERAGE_RANGE = (20, 100)


#################### MODULAR VARIANT SCORING METRIC CALCULATION ####################
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
def segregation_network_score(PedGraph, VariantEntry, mode, Scoring_Method= 'Original', categorical_scores=0, weights={'w_edge':0.6,'w_gen':0.2,'w_bet':0.2}, verbose= False):

    #Categorical Score Calculation
    if not categorical_scores:

        categorical_scores = raw_categorical_scoring(PedGraph,VariantEntry)

    edge_score = categorical_scores['edge_consistency']
    
    gen_score = max(0,min(categorical_scores['generation_continuity'],1))
    
    bet_score = categorical_scores['carrier_betweenness'] if mode=='AR' else 1-categorical_scores['carrier_betweenness']
    bet_score = max(0,min(bet_score,1))


    #Extended Categorical Scores
    if Scoring_Method == 'Extended':
        
        #when incorrect number of weights are given for extended scores, resort to default weights
        if len(weights.keys()) < 5:
            weights = {'w_edge': 0.6, 'w_gen': 0.1, 'w_bet': 0.1, 'w_found': 0.1, 'w_depth': 0.1}
        
        founder_score = categorical_scores['founder_influence']
        depth_score = categorical_scores['alt_depth']


    #Weighted Score Calculation
    total_score = (weights['w_edge'] * edge_score) + (weights['w_gen'] * gen_score) + (weights['w_bet'] * bet_score)
    if Scoring_Method == 'Extended':
        total_score += (weights['w_found'] * founder_score) + (weights['w_depth'] * depth_score)

    '''
    Current Scoring Metrics:
        Original:
            edge_score
            gen_score
            bet_score
        Extended:
            found_score
            depth_score
    '''
    if verbose:
        print(f"Edge Score: {edge_score}; Generational Score: {gen_score}; Betweeness Score: {bet_score}")
        if Scoring_Method == 'Extended':
            print(f"Founder Score: {founder_score}; Depth Score: {depth_score}")
        print(f'Segregation Score: {total_score}')
    return total_score














############## SEGREGATION SCORING WEIGHT OPTIMIZATION ##################

# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------
def max_score_highlighter(s):
    is_max = s == s.max()
    return [
        'background-color: green' if max_score and varID == 'chr1:100000_A>T'
        else 'background-color: red' if max_score
        else ''
        for varID, max_score in zip(s.index, is_max)]

def pprint_weights(weights_dict):
    for weight_name, weight_value in weights_dict.items():
        weight_value = round(weight_value, 3)
        print(f'{weight_name}: {weight_value}')
    print()

# ---------------------------------------------------------------------
# MARGIN-BASED WEIGHTS OPTIMIZATION OBJECTIVE
# ---------------------------------------------------------------------
def margin_weight_optimization_objective(weights_lst, Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode):

    weights_dict = {weight_names[i]: weights_lst[i] for i in range(len(weight_names))}
    margins = []
    for FamilyID, FamilyData in Multi_Ped_Dict.items():
        PedGraph, VarTable = FamilyData['PedGraph'], FamilyData['VarTable']
        CategoricalScores = FamilyData['CategoricalScores'][linked_variant]
        linked_score = segregation_network_score(PedGraph= PedGraph,
                                                 VariantEntry= VarTable[linked_variant],
                                                 mode= mode,
                                                 Scoring_Method= Scoring_Method,
                                                 weights= weights_dict,
                                                 categorical_scores= CategoricalScores)

        unlinked_scores = []
        for VarID, gt in VarTable.items():
            if VarID != linked_variant:
                CategoricalScores = FamilyData['CategoricalScores'][VarID]
                unlinked_score = segregation_network_score(PedGraph= PedGraph,
                                                            VariantEntry= VarTable[VarID],
                                                            mode= mode,
                                                            Scoring_Method= Scoring_Method,
                                                            weights= weights_dict,
                                                            categorical_scores= CategoricalScores)
                unlinked_scores.append(unlinked_score)

        max_unlinked_score = max(unlinked_scores)

        margin = linked_score - max_unlinked_score
        margins.append(margin)

    avg_margin = np.mean(margins)

    return 1 - avg_margin


# ---------------------------------------------------------------------
# RANK-BASED WEIGHTS OPTIMIZATION OBJECTIVE
# ---------------------------------------------------------------------
def rank_weight_optimization_objective(weights_lst, Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode):

    weights_dict = {weight_names[i]: weights_lst[i] for i in range(len(weight_names))}


    ranked_margins = []
    for FamilyID, FamilyData in Multi_Ped_Dict.items():
        PedGraph, VarTable = FamilyData['PedGraph'], FamilyData['VarTable']
        CategoricalScores = FamilyData['CategoricalScores'][linked_variant]
        linked_score = segregation_network_score(PedGraph= PedGraph,
                                                 VariantEntry= VarTable[linked_variant],
                                                 mode= mode,
                                                 Scoring_Method= Scoring_Method,
                                                 weights= weights_dict,
                                                 categorical_scores= CategoricalScores)

        unlinked_scores = []
        for VarID, gt in VarTable.items():
            if VarID != linked_variant:
                CategoricalScores = FamilyData['CategoricalScores'][VarID]
                unlinked_score  = segregation_network_score(PedGraph= PedGraph,
                                                            VariantEntry= VarTable[VarID],
                                                            mode= mode,
                                                            Scoring_Method= Scoring_Method,
                                                            weights= weights_dict,
                                                            categorical_scores= CategoricalScores)
                unlinked_scores.append(unlinked_score)

        all_scores = unlinked_scores + [linked_score]
        max_unlinked_score = max(unlinked_scores)

        all_scores.sort(reverse=True)
        linked_score_rank = all_scores.index(linked_score) + 1

        margin = linked_score - max_unlinked_score


        ranked_margins.append(linked_score_rank*margin)



    avg_ranked_margin = np.mean(ranked_margins)


    return len(VarTable) - avg_ranked_margin


# ---------------------------------------------------------------------
# WEIGHTS OPTIMIZATION OPERATIVE FUNCTION
# ---------------------------------------------------------------------
def weights_optimization(Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, Optimization_Method, initial_guess, mode= 'AD'):
    n_weights = len(weight_names)
    bounds = [(0.001,1)]*n_weights
    constraints = {'type': 'eq',
                  #figure out how this function is working
                  'fun': lambda w: np.sum(w)-1}

    if Optimization_Method == 'Margin':
        results = minimize(fun= margin_weight_optimization_objective,
                          x0= initial_guess,
                          args= (Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode),
                          bounds= bounds,
                          constraints= constraints)
    elif Optimization_Method == 'Rank':
        results = minimize(fun= rank_weight_optimization_objective,
                          x0= initial_guess,
                          args= (Multi_Ped_Dict, linked_variant, weight_names, Scoring_Method, mode),
                          bounds= bounds,
                          constraints= constraints)

    #Directly attaching weights with their names as dictionary for ease of use is scoring wrapper function
    optimized_weights = {weight_names[i]: results.x[i] for i in range(len(weight_names))}

    return optimized_weights


# ---------------------------------------------------------------------
# TRIAL-BASED WEIGHTS OPTIMIZATION
# ---------------------------------------------------------------------
#turn this into generative weights optimization only, and use standalone segregation scoring for real data application
def trial_based_segregation_scoring_weight_optimization(
                                                    #Segregation Scoring Parameters
                                                    Scoring_Method= 'Original',
                                                    weights= 0,
                                                    Optimization_Method= 'None',
                                                    Verbose= True,

                                                    #PedGraph Parameters
                                                    trial_count= TRIAL_COUNT,
                                                    Mode= 'AD',
                                                    generation_count= 3,
                                                    max_children = MAX_CHILDREN,
                                                    BackpropLikelihoodRange = BACKPROP_LIKELIHOOD_RANGE,
                                                    SpouseLikelihoodRange = SPOUSE_LIKELIHOOD_RANGE,
                                                    AffectedSpouse= AFFECTED_SPOUSE,


                                                    #VarTable Parameters
                                                    sequencing_coverage_range = SEQUENCE_COVERAGE_RANGE,
                                                    n_bg = 5,

                                                    #PedGraph and VarTable Parameters
                                                    alt_freq_range= ALT_FREQ_RANGE,
                                                    ):
    '''
    Takes multi-pedigree data dictionaries as input and outputs the dictionary with updated scores
    '''
    Multi_Ped_Dict = PedGraph_VarTable_generator(
                                            pedigree_count= trial_count,
                                            mode= Mode,
                                            max_children= max_children,
                                            generation_count= generation_count,
                                            sequencing_coverage_range= sequencing_coverage_range,
                                            BackpropLikelihoodRange= BackpropLikelihoodRange,
                                            SpouseLikelihoodRange= SpouseLikelihoodRange,
                                            AffectedSpouse= AffectedSpouse,
                                            
                                            n_bg = n_bg,
                                            
                                            alt_freq_range= alt_freq_range,)
    for FamID in Multi_Ped_Dict.keys():
        PedGraph = Multi_Ped_Dict[FamID]['PedGraph']
        VarTable = Multi_Ped_Dict[FamID]['VarTable']
        cat_score_dict = {}
        for VarID in VarTable.keys():
            cat_score_dict[VarID] = raw_categorical_scoring(G= PedGraph,
                                                            gt= VarTable[VarID])
        Multi_Ped_Dict[FamID]['CategoricalScores'] = cat_score_dict

    if Scoring_Method == 'Original':
        weight_names = ['w_edge', 'w_gen', 'w_bet']
    elif Scoring_Method == 'Extended':
        weight_names = ['w_edge', 'w_gen', 'w_bet', 'w_found', 'w_depth']

    #manually assignment of weights if no weights given if using original scoring
    if not weights:
        if Scoring_Method == 'Original':
            weights= {
                'w_edge': 0.6,
                'w_gen': 0.2,
                'w_bet': 0.2
            }
        elif Scoring_Method == 'Extended':
            weights= {
                'w_edge': 0.6,
                'w_gen': 0.1,
                'w_bet': 0.1,
                'w_found': 0.1,
                'w_depth': 0.1,
            }
        
    #Optimization of weights
    if Optimization_Method == 'Margin' or Optimization_Method == 'Rank':
        initial_guess = []
        for weight_name in weight_names:
            initial_guess.append(weights[weight_name])

        training_Multi_Ped_Dict = {}
        test_Multi_Ped_Dict = {}
        tt_split = 0.8
        for FamilyID in Multi_Ped_Dict.keys():
            if int(FamilyID[3:]) <= int(tt_split*len(Multi_Ped_Dict)):
                training_Multi_Ped_Dict[FamilyID] = Multi_Ped_Dict[FamilyID]
            else:
                test_Multi_Ped_Dict[FamilyID] = Multi_Ped_Dict[FamilyID]
        #Downsize the original multiple pedigree dict to the testing data now that we have done training testing split
        Multi_Ped_Dict = test_Multi_Ped_Dict
        weights= weights_optimization(Multi_Ped_Dict= training_Multi_Ped_Dict,
                                        linked_variant= 'chr1:100000_A>T',
                                        weight_names= weight_names,
                                        Scoring_Method= Scoring_Method,
                                        Optimization_Method= Optimization_Method,
                                        mode= Mode,
                                        initial_guess= initial_guess)


    


    All_Family_Score_df = pd.DataFrame(columns=Multi_Ped_Dict.keys())
    for FamilyID in Multi_Ped_Dict.keys():

        PedGraph, VarTable = Multi_Ped_Dict[FamilyID]['PedGraph'], Multi_Ped_Dict[FamilyID]['VarTable']


        CategoricalScores = Multi_Ped_Dict[FamilyID]['CategoricalScores']


        Multi_Ped_Dict[FamilyID][Scoring_Method] = {}
        for VarID in VarTable.keys():
            score = segregation_network_score(PedGraph= PedGraph,
                                                    VariantEntry= VarTable[VarID],
                                                    mode= Mode,
                                                    Scoring_Method= Scoring_Method,
                                                    weights= weights,
                                                    categorical_scores= CategoricalScores[VarID])
            Multi_Ped_Dict[FamilyID][Scoring_Method][VarID] = score


        All_Family_Score_df[FamilyID] = Multi_Ped_Dict[FamilyID][Scoring_Method]

    Correctly_Scored_Pedigrees = 0
    for FamilyID in Multi_Ped_Dict.keys():
        if max(Multi_Ped_Dict[FamilyID][Scoring_Method], key= Multi_Ped_Dict[FamilyID][Scoring_Method].get) == 'chr1:100000_A>T':
            Correctly_Scored_Pedigrees += 1
    Scoring_Method_Accuracy = Correctly_Scored_Pedigrees/len(Multi_Ped_Dict)

    #TODO figure out how to display variant score table in normal python script run
    if Verbose:
        print(f'{Scoring_Method} Segregation Scoring Results')
        styled_All_Family_Score_df = All_Family_Score_df.style.apply(max_score_highlighter, axis=0)
        styled_All_Family_Score_df
        print(f'{Mode} {Scoring_Method} Segregation Scoring Accuracy: {Scoring_Method_Accuracy}')
        print('Weights Used:')
        pprint_weights(weights)



    return Multi_Ped_Dict, weights, Scoring_Method_Accuracy










######################### SEGREGATION SCORING ##########################
def pedigree_segregation_scoring(Ped_Dict, Scoring_Method, Mode, Weights):

    PedGraph = Ped_Dict['PedGraph']
    VarTable = Ped_Dict['VarTable']

    CategoricalScores = {}
    for VarID in VarTable.keys():
        CategoricalScores[VarID] = raw_categorical_scoring(G= PedGraph,
                                                        gt= VarTable[VarID])
    Ped_Dict['CategoricalScores'] = CategoricalScores



    Ped_Dict[Scoring_Method] = {}
    for VarID in VarTable.keys():
        score = segregation_network_score(PedGraph= PedGraph,
                                                VariantEntry= VarTable[VarID],
                                                mode= Mode,
                                                Scoring_Method= Scoring_Method,
                                                weights= Weights,
                                                categorical_scores= CategoricalScores[VarID])
        Ped_Dict[Scoring_Method][VarID] = score

    return Ped_Dict