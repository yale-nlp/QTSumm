import evaluate
from autoacu import A3CU
from typing import List
from nltk import word_tokenize
from utils.tapas_acc import *

def get_sacrebleu_scores(predictions, references):
    sacrebleu = evaluate.load("sacrebleu")
    results = sacrebleu.compute(predictions=predictions, references=[[r] for r in references])
    return results["score"]

def get_rougel_scores(predictions, references):
    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predictions, references=references)
    return results["rougeL"]

def get_meteor_scores(predictions, references):
    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predictions, references=references)
    return results["meteor"]

def get_bert_scores(predictions, references):
    bertscore = evaluate.load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(len(results["f1"]))
    return sum(results["f1"]) / len(results["f1"])

def get_tapas_scores(prediction_file, dataset_name, split_name):
    tapas = TapasTest("google/tapas-large-finetuned-tabfact")
    data = MyData(prediction_file, dataset_name, split_name, tapas.tokenizer)
    test_dataloader = DataLoader(data, batch_size=16, shuffle=False, num_workers=4)
    results = tapas.test(test_dataloader)
    return results["acc"]
    
    
def get_autoacu_scores(predictions, references):
    a3cu = A3CU(device=0)
    recall_scores, prec_scores, f1_scores = a3cu.score(
        references=references,
        candidates=predictions,
        batch_size=32,
        output_path=None,
    )
    return sum(f1_scores) / len(f1_scores)

def get_prediction_lengths(predictions, references):
    total_length = 0
    for prediction in predictions:
        total_length += len(word_tokenize(prediction))
    return total_length / len(predictions)
    
    
def get_all_scores(predictions, references, prediction_file, dataset_name, split_name):
    all_scores = {}
    all_scores["sacreBLEU"] = get_sacrebleu_scores(predictions, references)
    print("sacreBLEU score: ", all_scores["sacreBLEU"])
    all_scores["Rouge-L"] = get_rougel_scores(predictions, references)
    print("Rouge-L score: ", all_scores["Rouge-L"])
    all_scores["METEOR"] = get_meteor_scores(predictions, references)
    print("METEOR score: ", all_scores["METEOR"])
    all_scores["BERTScore"] = get_bert_scores(predictions, references)
    print("BERTScore score: ", all_scores["BERTScore"])
    
    tapas_score = get_tapas_scores(prediction_file, dataset_name, split_name)
    all_scores["TAPAS"] = tapas_score
    print("TAPAS score: ", tapas_score)
    
    all_scores["AutoACU"] = get_autoacu_scores(predictions, references)
    print("AutoACU score: ", all_scores["AutoACU"])
    all_scores["Prediction Length"] = get_prediction_lengths(predictions, references)
    print("Prediction Length: ", all_scores["Prediction Length"])
    return all_scores