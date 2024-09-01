import pandas as pd
from rouge import Rouge

rouge = Rouge()

csv_file = 'MSc-data/output_summaries.csv'
df = pd.read_csv(csv_file)

rouge_scores_out = []

for index, row in df.iterrows():
    ref_summary = row['section_text']
    eval_summary_A = row['Model_A']
    eval_summary_B = row['Model_B']
    
    eval_1_rouge = rouge.get_scores(eval_summary_A, ref_summary)
    eval_2_rouge = rouge.get_scores(eval_summary_B, ref_summary)
    
    for metric in ["rouge-1", "rouge-2", "rouge-l"]:
        for label in ["f"]:
            eval_1_score = eval_1_rouge[0][metric][label]
            eval_2_score = eval_2_rouge[0][metric][label]
            
            if eval_1_score > eval_2_score:
                model_A_rank = 2
                model_B_rank = 3
            else:
                model_A_rank = 3
                model_B_rank = 2

            row = {
                "Index": index,
                "Metric": f"{metric} ({label})",
                "Summary A": eval_1_score,
                "Summary B": eval_2_score,
                "ref_Rank": 1,
                "model_A_Rank": model_A_rank,
                "model_B_Rank": model_B_rank
            }
            rouge_scores_out.append(row)


rouge_scores_df = pd.DataFrame(rouge_scores_out)
rouge_scores_df = rouge_scores_df.set_index(["Index", "Metric"])

output_csv_file = 'MSc-data/rouge_scores_output.csv'
rouge_scores_df.to_csv(output_csv_file)

