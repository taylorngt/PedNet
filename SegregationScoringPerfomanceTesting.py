import pandas as pd
from SegregationScoring import trial_based_segregation_scoring_weight_optimization
from pprint import pprint
import math
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon
# ---------------------------------------------------------------------
# BENCHMARKING CONFIGURATIONS
# ---------------------------------------------------------------------
NUMBER_TRIALS = 50
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
                                                                pedigree_count= NUMBER_PEDIGREES,
                                                                Scoring_Method= 'Original',
                                                                Optimization_Method= 'Rank',
                                                                Mode= mode,
                                                                generation_range= (3,3)
        )

        Top1s = []
        average_precisions = []
        for PedigreeID in test_results_dict.keys():
            manual_linked_score = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedScore']
            manual_linked_rank = test_results_dict[PedigreeID]['Original'][f'UnoptimizedLinkedRank']
            manual_linked_margin = test_results_dict[PedigreeID]['Original'][f'UnoptimizedMargin']
            manual_accuracy_metrics = compute_acc_metrics(rank_of_linked= manual_linked_rank)
            
            opt_linked_score = test_results_dict[PedigreeID]['Original'][f'RankLinkedScore']
            opt_linked_rank = test_results_dict[PedigreeID]['Original'][f'RankLinkedRank']
            opt_linked_margin = test_results_dict[PedigreeID]['Original'][f'RankMargin']
            opt_accuracy_metrics = compute_acc_metrics(rank_of_linked= opt_linked_rank)

            Top1s.append(opt_accuracy_metrics['Top1'])
            average_precisions.append(opt_accuracy_metrics['AP'])
            
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
        
        mean_AP = sum(average_precisions)/len(average_precisions)
        Top1Ratio = sum(Top1s)/len(Top1s)
        weights_results[mode].append({
            'TrialID': trial,

            'EdgeWeight': optimized_weights['w_edge'],
            'GenWeight' : optimized_weights['w_gen'],
            'BetweenessWeight': optimized_weights['w_bet'],

            'TopOneRankRatio' : Top1Ratio,
            'MeanAveragePrecision': mean_AP,
            })

# ---------------------------------------------------------------------
# BOX PLOTTING ACCURACY METRICS
# ---------------------------------------------------------------------
def plot_metric_comparison(metric, mode, results_df):
    plt.figure()

    df_box = pd.DataFrame({
        'manual' : results_df[f'manual_{metric}'],
        'optimized' : results_df[f'opt_{metric}']
    })

    df_box.boxplot()
    plt.title(f'{metric.upper()} Comparison')
    plt.ylabel(metric)
    plt.savefig(f'{mode}_{metric}_boxplot.png')
    plt.close()

for mode in MODES_OF_INHERITANCE:
    acc_df = pd.DataFrame(accuracy_results[mode])
    acc_df.to_csv(f'{mode}_benchmark_results.csv', index= False)
    print(f'\n{mode} benchmark results saved: {mode}_benchmark_results.csv')

    weights_df = pd.DataFrame(weights_results[mode])
    weights_df.to_csv(f'{mode}_weights_results.csv', index= False)
    print(f'{mode} benchmark weights saved: {mode}_weights_results.csv\n')

#     print('\nStatistical Testing (Paired t-test):')
#     for metric in ACC_METRICS:
#         plot_metric_comparison(metric, mode, results_df= df)
# # ---------------------------------------------------------------------
# # STATISTICAL TESTING
# # ---------------------------------------------------------------------     
#         t_stat, p_val = ttest_rel(
#                             df[f'manual_{metric}'],
#                             df[f'opt_{metric}'])
#         print(f'{metric.upper()}: t= {t_stat:.3f}, p= {p_val:.4f}')



