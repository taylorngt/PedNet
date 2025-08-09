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

    for gen_count in generation_counts:
        ss_results_df = pd.read_csv(results_log_path + f'/{gen_count}GenResults/{gen_count}Gen_Threshold_Results.csv')
        all_results_df = pd.concat(objs= [all_results_df, ss_results_df], ignore_index=True)
    
    os.makedirs(output_dir, exist_ok=True)

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
                    posthoc_df = pd.DataFrame(
                                    data=posthoc.summary().data[1:], 
                                    columns=posthoc.summary().data[0]
                                    )
                elif isinstance(posthoc, pd.DataFrame):
                    posthoc_df = posthoc
                else:
                    raise TypeError('Unexpected post hoc results')

                posthoc_df.to_csv(f'{output_dir}/MetricPostHocSummaries/post_hoc_threshold_{metric}.csv')
    

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
                    posthoc_df = pd.DataFrame(
                                    data=posthoc.summary().data[1:], 
                                    columns=posthoc.summary().data[0]
                                    )
                elif isinstance(posthoc, pd.DataFrame):
                    posthoc_df = posthoc
                else:
                    raise TypeError('Unexpected post hoc results')

                posthoc_df.to_csv(f'{output_dir}/PerformancePostHocSummaries/post_hoc_perfomance_{perf_metric}.csv')
    
    pd.DataFrame(stats_results).to_csv(f'{output_dir}/statistical_tests.csv', index=False)

    # -----------------------------
    # Visualizations
    # -----------------------------
    sns.set(style='whitegrid')

    #Threshold value boxplots
    
    for metric in metrics:
        plt.figure(figsize=(10,6))
        sns.violinplot(data= all_results_df, x= 'PedigreeSize', y= f'{metric} threshold')
        plt.title(f'{metric} Threshold Value by Pedigree Size')
        plt.savefig(f'{output_dir}/MetricViolinPlots/{metric}_theshold_by_size.png')
        plt.close()

    #Performance value boxplots
    for perf_metric in performance_metrics:
        plt.figure(figsize=(10,6))
        sns.violinplot(data= all_results_df, x= 'PedigreeSize', y= perf_metric)
        plt.title(f'{perf_metric} by Pedigree Size')
        plt.savefig(f'{output_dir}/PerformanceViolinPlots/{perf_metric}_by_size.png')
        plt.close()
    
    #AUC Heatmap
    # for metric in metrics:
    #     auc_matrix = all_results_df.groupby('PedigreeSize')[f'{metric} auc'].mean()
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(auc_matrix, annot=True, fmt=".2f", cmap="viridis")
    #     plt.title("Average {metric} AUC by Pedigree Size")
    #     plt.savefig(f'{output_dir}/{metric}_auc_heatmap.png')
    #     plt.close()

    print(f'Analysis complete. Results save to {output_dir}')

if __name__ == "__main__":
    run_stats_analysis(
            results_log_path= 'data/MoI_Benchmarking_Results',
            generation_counts= [3,4,5])
