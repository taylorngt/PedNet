import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon, shapiro
from os import makedirs



def run_tests(manual_metric, optimized_metric, label):
    diff = optimized_metric - manual_metric
    
    shapiro_stat, shapiro_p = shapiro(diff)
    normal = shapiro_p > 0.05

    if normal:
        stat, p_value = ttest_rel(optimized_metric, manual_metric)
        test_used = 'Paired t-test'
    else:
        stat, p_value = wilcoxon(optimized_metric, manual_metric)
        test_used = 'Wilcoxon Signed-Rank'
    
    return {
        "metric": label,
        "mean_manual": manual_metric.mean(),
        "mean_optimized": optimized_metric.mean(),
        "mean_diff": diff.mean(),
        "shapiro_p": shapiro_p,
        "normality_pass": normal,
        "test_used": test_used,
        "statistic": stat,
        "p_value": p_value
    }


def run_stats_analysis(result_log_path, output_dir):
    agg_results_df = pd.read_csv(result_log_path)
    
    performance_metrics = ['Top1','AP','NDCG']
    
    performance_results = []
    for perf_metric in performance_metrics:
        performance_results.append(run_tests(
                                        manual_metric= agg_results_df[f'AvgManual{perf_metric}'],
                                        optimized_metric= agg_results_df[f'AvgOpt{perf_metric}'],
                                        label= perf_metric))
    
    makedirs(output_dir, exist_ok=True)
    pd.DataFrame(performance_results).to_csv(f'{output_dir}/statistical_tests.csv', index= False)



    # -----------------------------
    # Visualizations
    # -----------------------------

    #slope plots
    for perf_metric in performance_metrics:
        plt.figure(figsize=(6,4))
        for _, row in agg_results_df.iterrows():
            plt.plot([0,1], [row[f"AvgManual{perf_metric}"], row[f"AvgOpt{perf_metric}"]],
                    marker="o", color="gray", alpha=0.7)
        plt.xticks([0,1], ["Default", "Optimized"])
        plt.ylabel(perf_metric)
        plt.title(f"{perf_metric} by Trial (Paired)")

        makedirs(f'{output_dir}/SlopePlots', exist_ok= True)
        plt.savefig(f'{output_dir}/SlopePlots/{perf_metric}_SlopePlot.png')
        plt.close()

    #diff plots
    for perf_metric in performance_metrics:
        diffs = agg_results_df[f'AvgOpt{perf_metric}'] - agg_results_df[f'AvgManual{perf_metric}']
        plt.figure(figsize=(5,4))
        sns.stripplot(x=diffs, color="blue", size=8)
        plt.axvline(0, color="black", linestyle="--")
        plt.xlabel(f"Improvement in {perf_metric} (Optimized - Default)")
        plt.title(f"Per-Trial Improvement in {perf_metric}")

        makedirs(f'{output_dir}/DiffPlots', exist_ok= True)
        plt.savefig(f'{output_dir}/DiffPlots/{perf_metric}_DiffPlot.png')
        plt.close()
    
    #paired boxplots
    for perf_metric in performance_metrics:
        melted = agg_results_df.melt(
                                    id_vars='TrialID',
                                    value_vars= [f'AvgManual{perf_metric}', f'AvgOpt{perf_metric}'],
                                    var_name='method', value_name=perf_metric)
        plt.figure(figsize=(5,4))
        sns.boxplot(x= 'method', y= perf_metric, data= melted)
        sns.swarmplot(x= 'method', y= perf_metric, data= melted, color='0.25')
        plt.title(f'{perf_metric} Across Trials')

        makedirs(f'{output_dir}/BoxPlots', exist_ok= True)
        plt.savefig(f'{output_dir}/BoxPlots/{perf_metric}_BoxPlot.png')
        plt.close()
    
if __name__ == '__main__':
    for mode in ['AD','AR']:
        run_stats_analysis(
            result_log_path= f'data/Segregation_Scoring_Trial_Results/{mode}_trial_results.csv',
            output_dir= f'data/Segregation_Scoring_Trial_Results/{mode}_results'
        )


