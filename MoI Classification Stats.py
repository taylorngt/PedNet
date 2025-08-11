import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, f_oneway, ttest_ind, kruskal, mannwhitneyu, chi2_contingency
import scikit_posthocs as sp
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os

def check_normality(groups, alpha=0.05):
    for g in groups:
        if len(g) < 3: #group too small for normality
            return False
        stat, p_value = shapiro(g)
        if p_value < alpha:
            return False
    return True


def run_numeric_tests(results_df, value_column, group_column):
    groups = [results_df.loc[results_df[group_column] == size, value_column].values
                for size in results_df[group_column].unique()]
    
    if len(groups) < 2:
        return None, None, None, None
    
    normality = check_normality(groups)

    if normality:
        #Parametric Test = ANOVA (difference between pedigree sizes)
        stat, p_value = f_oneway(*groups)
        posthoc = None
        #posthoc in the case the ANOVA is significant, difference between pedigree sizes
        if p_value < 0.05 and len(groups) > 2:
            posthoc = pairwise_tukeyhsd(endog= results_df[value_column], groups= results_df[group_column], alpha=0.05)
        return 'ANOVA', stat, p_value, posthoc

    else:
        #Nonparametric Test = Kruskal-Wallis
        stat, p_value = kruskal(*groups)
        posthoc= None
        if p_value < 0.05 and len(groups) > 2:
            posthoc = sp.posthoc_dunn(results_df, val_col=value_column, group_col=group_column, p_adjust='bonferroni')
        return "Kruskal-Wallis", stat, p_value, posthoc


def run_stats_analysis(results_log_path, generation_counts, output_dir='data/MoI_Benchmarking_Results'):

    all_results_df = pd.DataFrame()
    metrics = ['ratio_aff_parent', 'sibling_aff_ratio', 'gen_cov', 'avg_bet_unaff', 'aff_gen_clustering']

    averaged_thresholds =[]
    for gen_count in generation_counts:
        ss_results_df = pd.read_csv(results_log_path + f'/{gen_count}GenResults/{gen_count}Gen_Threshold_Results.csv')

        for metric in metrics:
           averaged_thresholds.append({
                'size':gen_count,
                'metric':metric,
                'threshold':np.mean(ss_results_df[f'{metric} threshold']),
                'direction':ss_results_df[f'{metric} direction'][0],
                'auc':np.mean(ss_results_df[f'{metric} auc'])
            })
        all_results_df = pd.concat(objs= [all_results_df, ss_results_df], ignore_index=True)


    os.makedirs(output_dir, exist_ok=True)
    #exporting averaged thresholds
    averaged_thresholds_df = pd.DataFrame(averaged_thresholds)
    averaged_thresholds_df.to_csv(f'{output_dir}/averaged_thresholds.csv', index= False)
    # required_columns = [
    #     'TrialID', 'PedigreeSize', 'accuracy', 'certainty', 'precision', 'recall', 'F1',
    #     'ratio_aff_parent threshold', 'ratio_aff_parent direction', 'ratio_aff_parent auc',
    #     'sibling_aff_ratio ', 'sibling_aff_ratio ', 'sibling_aff_ratio ', 'sibling_aff_ratio ',
    # ]
    
    metrics = ['ratio_aff_parent', 'sibling_aff_ratio', 'gen_cov', 'avg_bet_unaff', 'aff_gen_clustering']

    stats_results = []

    #Threshold values
    for metric in metrics:
        test_name, stat, p_value, posthoc = run_numeric_tests(
                                                results_df= all_results_df,
                                                value_column= f'{metric} threshold',
                                                group_column= 'PedigreeSize')
        
        if test_name:
            stats_results.append({
                            'test_type': test_name,
                            'metric': metric,
                            'statistic': stat,
                            'p_value':p_value
                            })
            if posthoc is not None:
                #Tukey's HSD returns a ruslts object with summary attribute
                if hasattr(posthoc, "summary"):
                    metric_posthoc_df = pd.DataFrame(
                                    data=posthoc.summary().data[1:], 
                                    columns=posthoc.summary().data[0]
                                    )
                elif isinstance(posthoc, pd.DataFrame):
                    metric_posthoc_df = posthoc
                else:
                    raise TypeError('Unexpected post hoc results')

                metric_posthoc_df.to_csv(f'{output_dir}/MetricPostHocSummaries/post_hoc_threshold_{metric}.csv')
    

    performance_metrics = ['accuracy', 'certainty', 'precision', 'recall', 'F1']
    for perf_metric in performance_metrics:
        test_name, stat, p_value, posthoc = run_numeric_tests(
                                                results_df= all_results_df,
                                                value_column= perf_metric,
                                                group_column= 'PedigreeSize')
        if test_name:
            stats_results.append({
                'test_type': test_name,
                'metric': perf_metric,
                'statistic': stat,
                'p_value':p_value
                })
            if posthoc is not None:
                #Tukey's HSD returns a ruslts object with summary attribute
                if hasattr(posthoc, "summary"):
                    perf_posthoc_df = pd.DataFrame(
                                    data=posthoc.summary().data[1:], 
                                    columns=posthoc.summary().data[0]
                                    )
                elif isinstance(posthoc, pd.DataFrame):
                    perf_posthoc_df = posthoc
                else:
                    raise TypeError('Unexpected post hoc results')

                perf_posthoc_df.to_csv(f'{output_dir}/PerformancePostHocSummaries/post_hoc_perfomance_{perf_metric}.csv')
    
    stats_results_df = pd.DataFrame(stats_results)
    stats_results_df.to_csv(f'{output_dir}/statistical_tests.csv', index=False)

    # -----------------------------
    # Visualizations
    # -----------------------------
    sns.set(style='whitegrid')
    metric_name_dict = {
        'ratio_aff_parent': 'Affected Parent-Child Pairs', 
        'sibling_aff_ratio': 'Affected Sibling Pairs', 
        'gen_cov': 'Generational Coverage', 
        'avg_bet_unaff': 'Average Betweenness Unaffected', 
        'aff_gen_clustering': 'Affected Generation Clustering'
    }

    perf_name_dict = {
        'accuracy': 'Accuracy', 
        'certainty': 'Certainty', 
        'precision': 'Precision', 
        'recall': 'Recall',
        'F1': 'F1'
    }

    #Threshold value violin plots
    metric_color_series = plt.color_sequences['tab20c']
    color_rotator = 0
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.violinplot(data= all_results_df, x= 'PedigreeSize', y= f'{metric} threshold', palette= metric_color_series[color_rotator:])

        plt.title(f'{metric_name_dict[metric]} Threshold Value by Pedigree Size')
        plt.ylabel(f'{metric_name_dict[metric]} Threshold')
        plt.savefig(f'{output_dir}/MetricViolinPlots/{metric}_theshold_by_size.png')
        plt.close()
        color_rotator += 4

    #Performance value violin plots
    perf_color_series = plt.color_sequences['tab20c'][8:]
    for perf_metric in performance_metrics:
        plt.figure(figsize=(10,6))
        sns.violinplot(data= all_results_df, x= 'PedigreeSize', y= perf_metric, palette=perf_color_series)
        plt.title(f'{perf_name_dict[perf_metric]} by Pedigree Size')
        plt.ylabel(f'{perf_name_dict[perf_metric]}')
        plt.xlabel('Pedigree Size')
        plt.ylim(top=1)
        plt.savefig(f'{output_dir}/PerformanceViolinPlots/{perf_metric}_by_size.png')
        plt.close()
    

    print(f'Analysis complete. Results save to {output_dir}')

if __name__ == "__main__":
    run_stats_analysis(
            results_log_path= 'data/MoI_Benchmarking_Results',
            generation_counts= [3,4,5])
