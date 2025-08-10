import pandas as pd
from SegregationScoring import trial_based_segregation_scoring_weight_optimization
from pprint import pprint
import math
import matplotlib.pyplot as plt
from os import makedirs
from scipy.stats import ttest_rel, wilcoxon
# ---------------------------------------------------------------------
# BENCHMARKING CONFIGURATIONS
# ---------------------------------------------------------------------
NUMBER_TRIALS = 40
NUMBER_PEDIGREES = 1000
ACC_METRICS = ['Top1', 'AP', 'NDCG']

# ---------------------------------------------------------------------
# TRIAL ACCURACY METRICS
# ---------------------------------------------------------------------
##### Hit at 1 #####
def calc_hit_at_1(rank_of_linked):
    '''
    Returns binary if the linked variant was correctly ranked at position 1 or not
    '''
    return 1.0 if rank_of_linked == 1 else 0.0

##### Average Precision #####
def calc_average_precision(rank_of_linked):
    '''
    Returns average precision based on the ranking of the linked variant
    '''
    return 1.0 / rank_of_linked

##### Normalized Discounted Cummulative Gain #####
def calc_NDCG(rank_of_linked):
    '''
    Returns NDCG based on the ranking of the linked variant
    '''
    return 1.0 / math.log2(rank_of_linked + 1)

##### Accuracy Metrics Calculation Wrapper #####
def compute_acc_metrics(rank_of_linked):
    return {
        'Rank' : rank_of_linked,
        'Top1' : calc_hit_at_1(rank_of_linked),
        'AP' : calc_average_precision(rank_of_linked),
        'NDCG' : calc_NDCG(rank_of_linked)
    }

# ---------------------------------------------------------------------
# SEGREGATION SCORING BY OPTIMIZATION AND SCORING METHOD
# ---------------------------------------------------------------------
MODES_OF_INHERITANCE = ['AD', 'AR']
Scoring_Methods = ['Original', 'Extended']
OPTIMIZATION_METHOD = 'Rank'


# ---------------------------------------------------------------------
# MAIN BENCHMARKING LOOP
# ---------------------------------------------------------------------
accuracy_results = {
    'AD' : [],
    'AR' : []
}
weights_results = {
    'AD' : [],
    'AR' : []
}

for trial in range(NUMBER_TRIALS):
    print(f'Currently Running: Trial #{trial+1}')
    for mode in MODES_OF_INHERITANCE:
        test_results_dict, optimized_weights = trial_based_segregation_scoring_weight_optimization(
                                                                pedigree_count= NUMBER_PEDIGREES,
                                                                Scoring_Method= 'Original',
                                                                Optimization_Method= OPTIMIZATION_METHOD,
                                                                Mode= mode,
                                                                generation_range= (3,4)
        )

        manual_Top1s = []
        manual_APs = []
        manual_NDCGs = []

        opt_Top1s = []
        opt_APs = []
        opt_NDCGs = []

        for PedigreeID in test_results_dict.keys():
            manual_linked_score = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedScore']
            manual_linked_rank = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedRank']
            manual_linked_margin = test_results_dict[PedigreeID]['Original'][f'UnoptimizedMargin']
            manual_accuracy_metrics = compute_acc_metrics(rank_of_linked= manual_linked_rank)

            manual_Top1s.append(manual_accuracy_metrics['Top1'])
            manual_APs.append(manual_accuracy_metrics['AP'])
            manual_NDCGs.append(manual_accuracy_metrics['NDCG'])
            

            opt_linked_score = test_results_dict[PedigreeID]['Original'][f'{OPTIMIZATION_METHOD}LinkedScore']
            opt_linked_rank = test_results_dict[PedigreeID]['Original'][f'{OPTIMIZATION_METHOD}LinkedRank']
            opt_linked_margin = test_results_dict[PedigreeID]['Original'][f'{OPTIMIZATION_METHOD}Margin']
            opt_accuracy_metrics = compute_acc_metrics(rank_of_linked= opt_linked_rank)

            opt_Top1s.append(opt_accuracy_metrics['Top1'])
            opt_APs.append(opt_accuracy_metrics['AP'])
            opt_NDCGs.append(opt_accuracy_metrics['NDCG'])
            
            accuracy_results[mode].append({
                'TrialID': trial+1,
                'PedigreeID': PedigreeID,

                'manual_Score' : manual_linked_score,
                'manual_Margin' : manual_linked_margin,
                'manual_Rank' : manual_accuracy_metrics['Rank'],
                'manual_Top1': manual_accuracy_metrics['Top1'],
                'manual_AP': manual_accuracy_metrics['AP'],
                'manual_NDCG': manual_accuracy_metrics['NDCG'],
                
                'opt_Score' : opt_linked_score,
                'opt_Margin' : opt_linked_margin,
                'opt_Rank' : opt_accuracy_metrics['Rank'],
                'opt_Top1': opt_accuracy_metrics['Top1'],
                'opt_AP': opt_accuracy_metrics['AP'],
                'opt_NDCG': opt_accuracy_metrics['NDCG'],
            })
        


        weights_results[mode].append({
            'TrialID': trial,

            'EdgeWeight': optimized_weights['w_edge'],
            'GenWeight' : optimized_weights['w_gen'],
            'BetweenessWeight': optimized_weights['w_bet'],

            'AvgManualTop1' : sum(manual_Top1s)/len(manual_Top1s),
            'AvgManualAP': sum(manual_APs)/len(manual_APs),
            'AvgManualNDCG': sum(manual_NDCGs)/len(manual_NDCGs),

            'AvgOptTop1' : sum(opt_Top1s)/len(opt_Top1s),
            'AvgOptAP': sum(opt_APs)/len(opt_APs),
            'AvgOptNDCG': sum(opt_NDCGs)/len(opt_NDCGs)
            })


for mode in MODES_OF_INHERITANCE:
    acc_df = pd.DataFrame(accuracy_results[mode])
    pedigree_results_dir = 'data/Pedigree_Scoring_Results'
    makedirs(pedigree_results_dir, exist_ok= True)
    acc_df.to_csv(f'{pedigree_results_dir}/{mode}_pedigree_results.csv', index= False)
    print(f'\n{mode} pedigree scoring results saved:') 
    print(f'{pedigree_results_dir}/{mode}_pedigree_results.csv')

    weights_df = pd.DataFrame(weights_results[mode])
    trial_results_dir = 'data/Segregation_Scoring_Trial_Results'
    makedirs(trial_results_dir, exist_ok= True)
    weights_df.to_csv(f'{trial_results_dir}/{mode}_trial_results.csv', index= False)
    print(f'{mode} scoring trial results saved:') 
    print(f'{trial_results_dir}/{mode}_trial_results.csv\n')



