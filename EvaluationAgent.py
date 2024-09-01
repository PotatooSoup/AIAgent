from openai import OpenAI
import os
import pandas as pd
import openai

openai_api_key = os.getenv('OPENAI_API_KEY')
csv_file = 'MSc-data/output_summaries.csv'
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<API KEY>"))

class ExtractionAgent:
    def __init__(self, csv_file, openai_api_key):
        self.df = pd.read_csv(csv_file)
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def extract_facts(self, text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": (
                    f"Please role-play as a medical expert, and extract the true facts from the original dialogue, "
                    f"reference summary, and the output facts from the following text:\n\n"
                    f"{text}\n\n"
                    f"Facts: A fact is defined as information that cannot be written in more than one sentence. "
                    f"For instance, the sentence 'The father died of stroke at age 89.' contains three facts: "
                    f"'The father died.', 'He was 89 years old.', and 'Stroke was the cause of death.' "
                    f"Ensure that each fact is clearly identified and accurately recorded for comparison and evaluation."
                )}
            ],
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0
        )
        facts = response.choices[0].message.content.strip().split('\n')
        return [fact.strip() for fact in facts if fact.strip()]
    
    def extract_all_facts(self):
        self.df['dialogue_facts'] = self.df['dialogue'].apply(lambda x: self.extract_facts(x))
        self.df['reference_facts'] = self.df['section_text'].apply(lambda x: self.extract_facts(x))
        self.df['modelA_facts'] = self.df['Model_A'].apply(lambda x: self.extract_facts(x))
        self.df['modelB_facts'] = self.df['Model_B'].apply(lambda x: self.extract_facts(x))
        return self.df
    
class AnnotationAgent:
    def __init__(self, facts_df, openai_api_key):
        self.df = facts_df
        self.client = openai.OpenAI(api_key=openai_api_key)
    
    def compare_facts_semantics(self, fact, text):
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": (
                    f"Compare the semantic meaning of the following fact with the provided text:\n\n"
                    f"Fact: {fact}\n\n"
                    f"Text: {text}\n\n"
                    f"Consider the fact to be semantically consistent with the text if they express the same or very similar information, even if the wording is different. "
                    f"For example, 'The father died of a stroke at age 89.' is semantically consistent with 'The father had a stroke and passed away at the age of 89.'.\n\n"
                    f"Please ensure that the correct facts in model A or B do not exceed the number of reference facts. Compare the facts from model A or B with the reference facts, and mark them as correct if they have the same semantic meaning. Otherwise, mark them as incorrect. Facts present in the reference but missing in the models should be marked as omitted. Do not fabricate data..  "
                    f"Semantic consistency (yes or no):"
                )}
            ],
            max_tokens=64,
            n=1,
            stop=None,
            temperature=0
        )
        semantic_consistency = response.choices[0].message.content.strip().lower()
        return 'yes' in semantic_consistency

    def calculate_facts(self, row):
        correct_facts = {'modelA': 0, 'modelB': 0}
        incorrect_facts = {'modelA': 0, 'modelB': 0}
        omit_facts = {'modelA': 0, 'modelB': 0}
        
        reference_text = ' '.join(row['reference_facts'])

        
        # Compare modelA facts with ref facts
        for fact in row['modelA_facts']:
            if self.compare_facts_semantics(fact, reference_text):
                correct_facts['modelA'] += 1
            else:
                incorrect_facts['modelA'] += 1
        
        # Compare modelB facts with ref facts
        for fact in row['modelB_facts']:
            if self.compare_facts_semantics(fact, reference_text):
                correct_facts['modelB'] += 1
            else:
                incorrect_facts['modelB'] += 1        
        
        # Ensure correct facts do not exceed reference facts
        correct_facts['modelA'] = min(correct_facts['modelA'], len(row['reference_facts']))
        correct_facts['modelB'] = min(correct_facts['modelB'], len(row['reference_facts']))
        
        # Calculate omit facts
        omit_facts['modelA'] = len(row['reference_facts']) - correct_facts['modelA']
        omit_facts['modelB'] = len(row['reference_facts']) - correct_facts['modelB']
        
        return pd.Series([
            len(row['dialogue_facts']),
            len(row['reference_facts']),
            len(row['modelA_facts']), correct_facts['modelA'], incorrect_facts['modelA'], omit_facts['modelA'],
            len(row['modelB_facts']), correct_facts['modelB'], incorrect_facts['modelB'], omit_facts['modelB']
        ])

    def evaluate(self):
        self.df[['num_dialogue_facts', 'num_reference_facts', 
                 'num_modelA_facts', 'num_modelA_correct_facts', 'num_modelA_incorrect_facts', 'num_modelA_omit_facts',
                 'num_modelB_facts', 'num_modelB_correct_facts', 'num_modelB_incorrect_facts', 'num_modelB_omit_facts']] = self.df.apply(self.calculate_facts, axis=1)
        
        return self.df[['num_dialogue_facts', 'num_reference_facts', 
                        'num_modelA_facts', 'num_modelA_correct_facts', 'num_modelA_incorrect_facts', 'num_modelA_omit_facts',
                        'num_modelB_facts', 'num_modelB_correct_facts', 'num_modelB_incorrect_facts', 'num_modelB_omit_facts']]


EVALUATION_PROMPT_TEMPLATE = """
You will be given three summaries written for the same article. Your task is to rank these summaries based on their fluency.
Please make sure you read and understand these instructions very carefully. 
Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:

{criteria}

Evaluation Steps:

{steps}

Example:

Source Text:

{document}

Summaries:

1. {summary1}
2. {summary2}
3. {summary3}

Evaluation Form:

- {metric_name}
Rank the summaries based on {metric_name}: 1, 2, 3. Please do not give me back any text.
Note: Please ensure that the response text only contains rankings in the expected forma, such as 1,2,3. 
"""

FLUENCY_SCORE_CRITERIA = """
Fluency evaluates the quality of the summary in terms of grammar, spelling, punctuation, word choice, and sentence structure.\
A good summary has few or no errors and is easy to read and follow.\
Please only return the number from (1-3) as the rank result.
"""

FLUENCY_SCORE_STEPS = """
Please rank these summaries from best to worst based on their fluency. \
Provide the ranks as a list of integers separated by commas (e.g., "1, 2, 3").
"""

CONCISENESS_SCORE_CRITERIA = """
Conciseness evaluate the summary should include only important information from the source document. \
The summary should include only important information from the source document. \
The summaries should not contain redundancies and excess information.\
Please only return the number from (1-3) as the rank result.
"""

CONCISENESS_SCORE_STEPS = """
1. Read the summary and the source dialogue carefully.
2. Compare the summary to the source dialogue and identify the main points of the article.
3. Assess how well the summary covers the main points of the article, and how much irrelevant or redundant information it contains.
4. Please rank these summaries from best to worst based on their fluency. Provide the ranks as a list of integers separated by commas (e.g., "1, 2, 3").
"""
   
class EvaluationAgent:
    def __init__(self, csv_file, annotation_results, openai_api_key):
        self.df = pd.read_csv(csv_file)
        self.annotation_results = annotation_results
        self.client = openai.OpenAI(api_key=openai_api_key)  


    def get_geval_score(self, criteria: str, steps: str, document: str, summary1: str, summary2: str, summary3: str, metric_name: str):
        prompt = EVALUATION_PROMPT_TEMPLATE.format(
            criteria=criteria,
            steps=steps,
            metric_name=metric_name,
            document=document,
            summary1=summary1,
            summary2=summary2,
            summary3=summary3,
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        ranks = response.choices[0].message.content.strip()
        return [int(rank) for rank in ranks.split(",")]

        
    def compare_matrix(self):
        results = []
        
        for index, row in self.df.iterrows():
            document = row['dialogue']
            ref_summary = row['section_text']
            gen_summaryA = row['Model_A']
            gen_summaryB = row['Model_B']
            
            # Fluency and Conciseness
            for eval_type, (criteria, steps) in evaluation_metrics.items():
                ranks = self.get_geval_score(criteria, steps, document, ref_summary, gen_summaryA, gen_summaryB, eval_type)
                results.append({
                    'item_index': index,
                    'Evaluation Type': eval_type,
                    'Reference_rank': ranks[0],
                    'ModelA_rank': ranks[1],
                    'ModelB_rank': ranks[2]
                })
            
            # Consistency
            num_correct_facts_A = self.annotation_results['num_modelA_correct_facts'][index]
            num_total_facts_A = self.annotation_results['num_modelA_facts'][index]
            num_correct_facts_B = self.annotation_results['num_modelB_correct_facts'][index]
            num_total_facts_B = self.annotation_results['num_modelB_facts'][index]
            
            consistency_A = num_correct_facts_A / num_total_facts_A if num_total_facts_A > 0 else 0
            consistency_B = num_correct_facts_B / num_total_facts_B if num_total_facts_B > 0 else 0
            
            if consistency_A > consistency_B:
                consistency_ranks = [2, 3]
            elif consistency_A < consistency_B:
                consistency_ranks = [3, 2]
            else:
                consistency_ranks = [2, 2]
            
            results.append({
                'item_index': index,
                'Evaluation Type': 'Consistency',
                'Reference_rank': 1,
                'ModelA_rank': consistency_ranks[0],
                'ModelB_rank': consistency_ranks[1]
            })
            
            # Comprehensive
            num_omit_facts_A = self.annotation_results['num_modelA_omit_facts'][index]
            num_dialogue_facts = self.annotation_results['num_dialogue_facts'][index]
            omission_rate_A = 1 - (num_omit_facts_A / num_dialogue_facts)
            
            num_omit_facts_B = self.annotation_results['num_modelB_omit_facts'][index]
            omission_rate_B = 1 - (num_omit_facts_B / num_dialogue_facts)
            
            if omission_rate_A > omission_rate_B:
                comprehensive_ranks = [2, 3]
            elif omission_rate_A < omission_rate_B:
                comprehensive_ranks = [3, 2]
            else:
                comprehensive_ranks = [2, 2]
            
            results.append({
                'item_index': index,
                'Evaluation Type': 'Comprehensive',
                'Reference_rank': 1,
                'ModelA_rank': comprehensive_ranks[0],
                'ModelB_rank': comprehensive_ranks[1]
            })
            
        return results
    

extraction_agent = ExtractionAgent(csv_file, openai_api_key)
facts_df = extraction_agent.extract_all_facts()
annotation_agent = AnnotationAgent(facts_df, openai_api_key)
annotation_results = annotation_agent.evaluate()
evaluation_metrics = {
    "Fluency": (FLUENCY_SCORE_CRITERIA, FLUENCY_SCORE_STEPS),
    "Conciseness": (CONCISENESS_SCORE_CRITERIA, CONCISENESS_SCORE_STEPS),
}

evaluation_agent = EvaluationAgent(csv_file, annotation_results, openai.api_key)
sorted_results = evaluation_agent.compare_matrix()
df_results = pd.DataFrame(sorted_results)
output_csv_file = 'MSc-data/Evaluation_results.csv'
df_results.to_csv(output_csv_file, index=False)