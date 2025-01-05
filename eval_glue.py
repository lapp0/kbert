from evaluate.evaluation_suite import EvaluationSuite, SubTask
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


class GLUESuite(EvaluationSuite):
    def __init__(self):
        super().__init__("glue eval suite")
        self.suite = [
            # CoLA
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="cola",
                split="validation[:10]",
                args_for_task={
                    "metric": "matthews_correlation",
                    "input_column": "sentence",
                    "label_column": "label",
                }
            ),
            # SST-2
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="sst2",
                split="validation[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "sentence",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0.0, "LABEL_1": 1.0}
                }
            ),
            # MRPC
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mrpc",
                split="validation[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                }
            ),
            # TODO: implement regression
            # STS-B
            # SubTask(
            #     task_type="regression",
            #     data="glue",
            #     subset="stsb",
            #     split="validation[:10]",
            #     args_for_task={
            #         "metric": "spearman",
            #         "input_column": "sentence1",
            #         "second_input_column": "sentence2",
            #         "label_column": "label",
            #     }
            # ),
            # QQP
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qqp",
                split="validation[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "question1",
                    "second_input_column": "question2",
                    "label_column": "label",
                }
            ),
            # MNLI (matched)
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mnli",
                split="validation_matched[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "label_column": "label",
                }
            ),
            # MNLI (mismatched)
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="mnli",
                split="validation_mismatched[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "premise",
                    "second_input_column": "hypothesis",
                    "label_column": "label",
                }
            ),
            # QNLI
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="qnli",
                split="validation[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "question",
                    "second_input_column": "sentence",
                    "label_column": "label",
                }
            ),
            # RTE
            SubTask(
                task_type="text-classification",
                data="glue",
                subset="rte",
                split="validation[:10]",
                args_for_task={
                    "metric": "accuracy",
                    "input_column": "sentence1",
                    "second_input_column": "sentence2",
                    "label_column": "label",
                    "label_mapping": {"LABEL_0": 0.0, "LABEL_1": 1.0}
                }
            )
        ]


def modernbert_pipe(model_uri="answerdotai/ModernBERT-base"):
    model = AutoModelForSequenceClassification.from_pretrained(model_uri)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    return pipeline(task="text-classification", model=model, tokenizer=tokenizer)


def debertav3_pipe(model_uri="microsoft/deberta-v3-base"):
    model = AutoModelForSequenceClassification.from_pretrained(model_uri)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
    return pipeline(task="text-classification", model=model, tokenizer=tokenizer)


def kbert_pipe(model_uri):
    from model import KBERTForSequenceClassification
    model = KBERTForSequenceClassification.from_pretrained(model_uri)
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")  # kbert uses modernbert tokenizer
    return pipeline(task="text-classification", model=model, tokenizer=tokenizer)


if __name__ == "__main__":
    results = GLUESuite().run(modernbert_pipe())
    print("Evaluation Results:", results)
