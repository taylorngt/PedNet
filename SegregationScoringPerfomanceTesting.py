import pandas as pd
from SegregationScoring import trial_based_segregation_scoring_weight_optimization
from pprint import pprint
import math
# ---------------------------------------------------------------------
# BENCHMARKING CONFIGURATIONS
# ---------------------------------------------------------------------
NUMBER_TRIALS = 10
NUMBER_PEDIGREES = 50
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
Optimization_Methods = ['None', 'Rank']


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
    for mode in MODES_OF_INHERITANCE:
        test_results_dict, optimized_weights = trial_based_segregation_scoring_weight_optimization(
                                                                trial_count= NUMBER_PEDIGREES,
                                                                Scoring_Method= 'Original',
                                                                Optimization_Method= 'Rank',
                                                                Mode= mode,
                                                                generation_count= 3
        )

        weights_results[mode].append(optimized_weights)

        for PedigreeID in test_results_dict.keys():
            manual_linked_rank = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedRank']
            manual_accuracy_metrics = compute_acc_metrics(rank_of_linked= manual_linked_rank)
            
            opt_linked_rank = test_results_dict[PedigreeID]['Original'][f'RankLinkedRank']
            opt_accuracy_metrics = compute_acc_metrics(rank_of_linked= opt_linked_rank)
            
            accuracy_results[mode].append({
                'TrialID': trial,
                'PedigreeID': PedigreeID,

                'manual_rank' : manual_accuracy_metrics['Rank'],
                'manual_top1': manual_accuracy_metrics['Top1'],
                'manual_ap': manual_accuracy_metrics['AP'],
                'manual_ndcg': manual_accuracy_metrics['NDCG'],
                
                'opt_rank' : opt_accuracy_metrics['Rank'],
                'opt_top1': opt_accuracy_metrics['Top1'],
                'opt_ap': opt_accuracy_metrics['AP'],
                'opt_ndcg': opt_accuracy_metrics['NDCG'],
            })

for mode in MODES_OF_INHERITANCE:
    df = pd.DataFrame(accuracy_results[mode])
    df.to_csv(f'{mode}_benchmark_results.csv', index= False)
    print(f'{mode} results saved: {mode}_benchmark_results.csv')
