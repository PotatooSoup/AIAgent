# MSc-Project
This repository contains the data and source code for the MSc project "Development of LLM- based Agents for Clinical Dialogue Summarisation and Evaluation."

## Description
This study evaluates the performance of large language models (LLMs) in summarising doctor-patient consultations and compares their quality against standard automatic and manual evaluation metrics. The study aims to develop an AI agent leveraging LLMs, specifically ChatGPT-4omini and Qwen2-72B-Instruct, to evaluate the performance of automatic summarisation in clinical contexts.


The study overview:  
<div align="center"><img width="350" height="400" src="https://github.com/PotatooSoup/AIAgent/blob/main/image/workflow.png"/></div>


The Agent pipline:  
<div align="center"><img width="450" height="400" src="https://github.com/PotatooSoup/AIAgent/blob/main/image/agent.png"/></div>


## Requirements
python 3.8 and above  
openai 1.37.2

# Datasets, Code and Evaluation Questionnaire

## Main Dataset
The MTS-Dialog dataset is a new collection of 1.7k short doctor-patient conversations and corresponding summaries (section headers and contents).
The training set consists of 1,201 pairs of conversations and associated summaries.

The validation set consists of 100 pairs of conversations and their summaries.

MTS-Dialog includes 2 test sets; each test set consists of 200 conversations and associated section headers and contents.

The full list of normalized section headers:
```
    1. fam/sochx [FAMILY HISTORY/SOCIAL HISTORY]
    2. genhx [HISTORY of PRESENT ILLNESS]
    3. pastmedicalhx [PAST MEDICAL HISTORY]
    4. cc [CHIEF COMPLAINT]
    5. pastsurgical [PAST SURGICAL HISTORY]
    6. allergy
    7. ros [REVIEW OF SYSTEMS]
    8. medications
    9. assessment
    10. exam
    11. diagnosis
    12. disposition
    13. plan
    14. edcourse [EMERGENCY DEPARTMENT COURSE]
    15. immunizations
    16. imaging
    17. gynhx [GYNECOLOGIC HISTORY]
    18. procedures
    19. other_history
    20. labs
```
In this study we used 20 history of present illness data from the test set as input.

## Source Code
The source code for the summarisation and evaluation of doctor-patient consultations.

## Manual Evaluation Questionnaire
The human evaluation questionnaire example can be found at ["Rank the Summary"](https://forms.office.com/Pages/ResponsePage.aspx?id=KVxybjp2UE-B8i4lTwEzyNeZ6CsxK85FpPqFov_M7hBUN0xON1cwQ0JDUkNZTjhMVUVOOU1UV0Y3Ny4u) 
We have sent out 10 questionnaires contain 10 different summary results for human expert to evaluate.


