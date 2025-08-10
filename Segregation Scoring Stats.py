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


def run_stats_analysis(result_log_path, output_dir, mode):
    agg_results_df = pd.read_csv(result_log_path)
    
    performance_metrics = ['Top1','AP','NDCG']
    
    performance_results = []
    for perf_metric in performance_metrics:
        performance_results.append(run_tests(
                                        manual_metric= agg_results_df[f'AvgManual{perf_metric}'],
                                        optimized_metric= agg_results_df[f'AvgOpt{perf_metric}'],
                                        label= perf_metric))
    
    makedirs(output_dir, exist_ok=True)
    performance_results_df = pd.DataFrame(performance_results)
    performance_results_df.to_csv(f'{output_dir}/statistical_tests.csv', index= False)



    # -----------------------------
    # Visualizations
    # -----------------------------
    metric_colors = {
        "AP": "#1f77b4",     # blue
        "NDCG": "#2ca02c",   # green
        "Top1": "#d62728"   # red
    }   
    
    #slope plots
    # for perf_metric in performance_metrics:
    #     plt.figure(figsize=(6,4))
    #     for _, row in agg_results_df.iterrows():
    #         plt.plot([0,1], [row[f"AvgManual{perf_metric}"], row[f"AvgOpt{perf_metric}"]],
    #                 marker="o", color="gray", alpha=0.7)
    #     plt.xticks([0,1], ["Default", "Optimized"])
    #     plt.ylabel(perf_metric)
    #     plt.title(f"{perf_metric} by Trial (Paired)")

    #     makedirs(f'{output_dir}/SlopePlots', exist_ok= True)
    #     plt.savefig(f'{output_dir}/SlopePlots/{perf_metric}_SlopePlot.png')
    #     plt.close()

    # #diff plots
    # for perf_metric in performance_metrics:
    #     diffs = agg_results_df[f'AvgOpt{perf_metric}'] - agg_results_df[f'AvgManual{perf_metric}']
    #     plt.figure(figsize=(5,4))
    #     sns.stripplot(x=diffs, color="blue", size=8)
    #     plt.axvline(0, color="black", linestyle="--")
    #     plt.xlabel(f"Improvement in {perf_metric} (Optimized - Default)")
    #     plt.title(f"Per-Trial Improvement in {perf_metric}")

    #     makedirs(f'{output_dir}/DiffPlots', exist_ok= True)
    #     plt.savefig(f'{output_dir}/DiffPlots/{perf_metric}_DiffPlot.png')
    #     plt.close()
    
    #weights box plot
    edge_weights = agg_results_df['EdgeWeight']
    gen_weights = agg_results_df['GenWeight']
    bet_weights = agg_results_df['BetweenessWeight']
    fig = plt.figure(figsize=(7,4))
    ax = fig.add_subplot(111)

    sns.boxplot(data=[edge_weights,gen_weights,bet_weights], palette=plt.color_sequences['tab20'], legend = False)

    plt.title(f'{mode} Optimized Weights')
    ax.set_xticklabels(['Edge Consistency', 'Generation Continuity', 'Carrier Betweenness'])
    makedirs(f'{output_dir}/WeightBoxPlots', exist_ok= True)
    plt.savefig(f'{output_dir}/WeightBoxPlots/{perf_metric}_WeightsPlot.png')
    plt.close()

    color_seq = plt.color_sequences['tab20'][:2] if mode == 'AD' else plt.color_sequences['tab20b'][:2]
    #paired violin
    for perf_metric in performance_metrics:
        unopt_metric = agg_results_df[f'AvgManual{perf_metric}']
        opt_metric = agg_results_df[f'AvgOpt{perf_metric}']
        
        fig = plt.figure(figsize=(5,4))
        ax = fig.add_subplot(111)
        #color = metric_colors[perf_metric]

        sns.violinplot(data= [unopt_metric, opt_metric], legend= False, palette=color_seq)

        # Add significance marker
        y_max = np.concatenate((unopt_metric,opt_metric)).max()
        y_sig = y_max*0.02 + y_max
        x1, x2 = 0, 1
        plt.plot([x1, x1, x2, x2], [y_sig, y_sig+0.005, y_sig+0.005, y_sig], color='black')
        
        for _, row in performance_results_df.iterrows():
            if row['metric'] == perf_metric:
                p_value = row['p_value']

        if p_value < 0.001:
            sig_label = "***"
        elif p_value < 0.01:
            sig_label = "**"
        elif p_value < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"
            
        plt.text((x1+x2)/2, y_sig+0.008, sig_label, ha='center', va='bottom', fontsize=12)
        plt.title(f'Average {perf_metric} Across {mode} Trials')
        ax.set_xticklabels(['Unoptimize', 'Optimized'])
        plt.ylim(top = 1)
        makedirs(f'{output_dir}/ViolinPlots', exist_ok= True)
        plt.savefig(f'{output_dir}/ViolinPlots/{perf_metric}_ViolinPlot.png')
        plt.close()
    
if __name__ == '__main__':
    for mode in ['AD','AR']:
        run_stats_analysis(
            result_log_path= f'data/Segregation_Scoring_Trial_Results/{mode}_trial_results.csv',
            output_dir= f'data/Segregation_Scoring_Trial_Results/{mode}_results',
            mode= mode
        )


