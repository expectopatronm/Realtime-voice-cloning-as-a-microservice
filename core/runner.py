import json
import itertools

from core.qg_qae.qg import QuestionGenerator
from core.qg_qae.qae import QuestionAnswerEvaluator
from core.qg_qae.utils import extract_phrase, remove_duplicates

import torch

from sentence_transformers import SentenceTransformer, util

from transformers import (
    DistilBertForQuestionAnswering,
    DistilBertTokenizerFast,
)


question_generator = QuestionGenerator()
question_answer_evaluator = QuestionAnswerEvaluator()

vectorizer_model = SentenceTransformer("distiluse-base-multilingual-cased")
question_answering_model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased"
)
question_answering_model.load_state_dict(
    torch.load("core/qa/saved_models/model_weights.pth")
)

question_answering_tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)


def faq_dataset_generator(passage):
    """Generate Questions for a Passage."""

    answers = extract_phrase(passage, "NP")
    answers.extend(extract_phrase(passage, "VP"))

    questions = question_generator.generate_questions(answers, passage)
    questions = remove_duplicates(questions)

    encoded_qa_pairs = question_answer_evaluator.encode_qa_pairs(
        questions, answers
    )
    scores = question_answer_evaluator.get_scores(encoded_qa_pairs)
    q_a_score_dict = dict(zip(scores, questions))

    if len(q_a_score_dict) <= 10:
        return list(dict(sorted(q_a_score_dict.items())).values())
    else:
        return list(
            dict(itertools.islice(sorted(q_a_score_dict.items()), 10)).values()
        )


def faq_inference_runner(input_text):
    """Infer Answer for an input Question."""

    input_text_embedding = vectorizer_model.encode(
        input_text, convert_to_tensor=True
    )

    faq_corpus_dict = {}
    with open("core/static/faq-corpus.json") as json_file:
        faq_corpus_dict = json.load(json_file)

    max_dict = {}
    for key, questions in faq_corpus_dict.items():
        max_dict[key] = max(
            [
                util.pytorch_cos_sim(
                    input_text_embedding,
                    vectorizer_model.encode(question, convert_to_tensor=True),
                ).item()
                for question in questions
            ]
        )
        max_key = max(max_dict, key=max_dict.get)
    return max_key


def squad_style_dataset_generator(passage):
    """Generate SQUAD Style Dataset for a Passage."""

    # answers = extract_phrase(passage, 'NP')
    answers = extract_phrase(passage, "VP")

    questions = question_generator.generate_questions(answers, passage)

    encoded_qa_pairs = question_answer_evaluator.encode_qa_pairs(
        questions, answers
    )
    scores = question_answer_evaluator.get_scores(encoded_qa_pairs)

    answers_dict = dict(zip(scores, answers))
    questions_dict = dict(zip(scores, questions))

    q_a_dict = {}
    for key, value in questions_dict.items():
        if value not in q_a_dict:
            q_a_dict[value] = [key]
        else:
            q_a_dict[value].append(key)

    for key in q_a_dict:
        q_a_dict[key] = [answers_dict[score] for score in q_a_dict[key]]
        q_a_dict[key].append(
            {
                "score": list(questions_dict.keys())[
                    list(questions_dict.values()).index(key)
                ]
            }
        )

    i = 0
    dataset = []
    for key in q_a_dict:
        dataset.append(
            {
                "context": passage.replace("\n", ""),
                "qas": [
                    {
                        "answers": [
                            {"answer_start": "", "text": answer}
                            for answer in q_a_dict[key][:-1]
                        ],
                        "id": i,
                        "question": key,
                    }
                ],
            }
        )

    return dataset


def squad_style_inference_runner(question, context):
    """Infer Answer for an input Question given a Context."""

    inputs = question_answering_tokenizer(
        question, context, add_special_tokens=True, return_tensors="pt"
    )
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = question_answering_model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = question_answering_tokenizer.convert_tokens_to_string(
        question_answering_tokenizer.convert_ids_to_tokens(
            input_ids[answer_start:answer_end]
        )
    )

    return answer, (answer_start.item(), answer_end.item())
