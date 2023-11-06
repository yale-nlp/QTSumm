import pandas as pd
import abc
import math
import random
from typing import Dict, List, List
from transformers import AutoTokenizer, BasicTokenizer

random.seed(42)

class TableLinearize(abc.ABC):

    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        pass


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + " "
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "col : " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + str(row_index) + " : " + row_str

class TableTruncate(abc.ABC):

    def __init__(self, tokenizer: BasicTokenizer = None, max_input_length: int = 1024):
        """
        The class `TableTruncate` is used to compress a table to fit in memory.
        :param tokenizer: a huggingface transformer's tokenizer, to be used on BPE encoding to estimate expected tokens
        :param max_input_length: the maximum length of `question` and `table`, i.e., the max position id of a model
        """
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.max_length = max_input_length

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        """
        Given a table, return a truncated table with the same format.
        We enable optionally providing question and answer for precise truncating.
        :return: no return value, but may modify table_content and answer
        """
        pass


class CellLimitTruncate(TableTruncate):
    """
    Limit the maximum length of cell values in a table to truncate the overall length
    """

    def __init__(self, max_cell_length: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.max_cell_length = max_cell_length

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        cell_mapping = {}
        for row in table_content["rows"]:
            for i, cell in enumerate(row):
                truncate_cell = self.truncate_cell(cell)
                if truncate_cell is not None:
                    cell_mapping[cell] = truncate_cell
                    row[i] = truncate_cell

        # modify the answer list
        for i, case in enumerate(answer):
            if case in cell_mapping.keys():
                answer[i] = cell_mapping[case]

    def truncate_cell(self, cell_value):
        # do not process on these cases
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        if cell_value.strip() != "":
            try_tokens = self.tokenizer.tokenize(cell_value)
            if len(try_tokens) >= self.max_cell_length:
                retain_tokens = try_tokens[:self.max_cell_length]
                retain_cell_value = self.tokenizer.convert_tokens_to_string(retain_tokens)
                return retain_cell_value
            else:
                return None
        else:
            return cell_value


class RowDeleteTruncate(TableTruncate):
    """
    The row deleting principle is straightforward: randomly deleting rows to fit the table into memory,
    but do not make it too small (e.g., just lower than the limitation is ok).
    """

    def __init__(self, table_linearize: TableLinearize, **kwargs):
        super().__init__(**kwargs)
        self.table_linearize = table_linearize

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        """
        :param table_content: {"header": xxx, "rows": xxx, "id" (Optionally): xxx}
        :param question: natural language sentence
        :param answer: if for training, is the supervision; otherwise will be empty
        """
        delete_ratio, remain_token_len = self.estimate_delete_ratio(table_content, question)
        # randomly delete unrelated rows
        self.delete_unrealted_rows(table_content, question, answer, delete_ratio)
        # guarantee the result < self.max_length
        maximum_keep_rows = 0
        for ind, row_example in enumerate(table_content["rows"]):
            value_string = self.table_linearize.process_row(row_example, ind + 1)
            value_token_len = len(self.tokenizer.tokenize(value_string))
            # over the size limit, and take action
            if value_token_len > remain_token_len:
                break
            remain_token_len -= value_token_len
            maximum_keep_rows += 1
        del table_content["rows"][maximum_keep_rows:]

    def estimate_delete_ratio(self, table_content: Dict, question: str):
        assert "header" in table_content and "rows" in table_content
        number_of_rows = len(table_content["rows"])
        # calculate the tokens of header, special tokens will only be pre-prepended into question
        question_tokens = self.tokenizer.tokenize(question, add_special_tokens=True)
        # calculate the tokens of header
        header_string = self.table_linearize.process_header(table_content["header"])
        header_tokens = self.tokenizer.tokenize(header_string, add_special_tokens=False)
        # split all cell values into tokens and see how many can be accommodated
        used_token_len = len(question_tokens) + len(header_tokens)
        # remaining token space for rows
        remain_token_len = self.max_length - used_token_len

        value_string = ""
        for _, row_example in enumerate(table_content["rows"]):
            # use a general index to roughly estimate the overall token len
            value_string += self.table_linearize.process_row(row_example, 100) + " "
        value_token_len = len(self.tokenizer.tokenize(value_string))

        if value_token_len < remain_token_len:
            # no row will be deleted
            return 0.0, remain_token_len
        else:
            # calc a roughly delete rate
            return 1.0 - remain_token_len / value_token_len, remain_token_len

    def delete_unrealted_rows(self, table_content: Dict, question: str, answer: List, delete_ratio: float):
        """
        The argument answer is used only during training.
        """
        truncated_unrelated_indices = []
        related_indices = []
        if len(answer) == 0:
            answer_set = set([])
        else:
            answer_set = set([ans_ex.lower() for ans_ex in answer])
        # add question key words into answer set
        if question is not None:
            answer_set.update(question.split())
        question_set = set(question.strip("?!.,").split(" "))
        row_max_len = len(table_content["rows"])
        for _row_idx, row in enumerate(table_content["rows"]):
            lower_row = set([str(cell).lower() for cell in row])
            if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
                truncated_unrelated_indices.append(_row_idx)
            else:
                # add neighbours to preserve information aggressively
                related_indices.extend([_row_idx - 2, _row_idx - 1,
                                        _row_idx,
                                        _row_idx + 1, _row_idx + 2])

        # remove the neighbours
        truncated_unrelated_indices = [_row_idx for _row_idx in truncated_unrelated_indices
                                       if _row_idx not in related_indices]
        # select some cases to drop
        drop_items = min(len(truncated_unrelated_indices), int(len(table_content["rows"]) * delete_ratio))
        drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)

        for _row_idx in reversed(range(row_max_len)):
            if _row_idx in drop_row_indices:
                del table_content["rows"][_row_idx]

        # only when the drop ratio is too large, logging for warning.
        if "id" in table_content and len(drop_row_indices) > 0:
            logger.warning("Delete {:.2f} rows in table {}".format(len(drop_row_indices), table_content["id"]))

class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 target_delimiter: str = ", "):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.target_delimiter = target_delimiter

    def process_input(self, table_content: Dict, question: str, answer: List[str]) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """
        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            truncate_func.truncate_table(table_content, question, answer)
        # linearize a table into a string
        linear_table = self.table_linearize_func.process_table(table_content)
        # concat question with linear_table
        joint_input = question + " " + linear_table
        return joint_input

    def process_output(self, answer: List[str]) -> str:
        """
        Flatten the output for translation
        """
        if type(answer) == str:
            output = answer
        else:
            output = self.target_delimiter.join(answer)
            
        if output.strip() == "":
            raise Exception("The Answer is EMPTY!")
        else:
            return output


def get_default_processor(max_cell_length, max_input_length, model_name_or_path):
    table_linearize_func = IndexedRowTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path),
                          max_input_length=max_input_length),
        RowDeleteTruncate(table_linearize=table_linearize_func,
                          max_input_length=max_input_length)
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs)
    return processor

def preprocess_qtsumm_function(examples, is_training=False, tokenizer=None, data_args=None, model_args=None, config=None):
    """
    The is_training FLAG is used to identify if we could use the supervision
    to truncate the table content if it is required.
    """
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=data_args.max_source_length, model_name_or_path=model_args.model_name_or_path)
    titles = [f"{table['title'].lower()}" for table in examples["table"]]
    example_tables = examples["table"]
    questions = [question.lower() for question in examples["query"]]

    queries = [f"Question: {question} Title: {title}" for question, title in zip(questions, titles)]

    tables = []
    for example_table in example_tables:
        column = example_table["header"]
        rows = example_table["rows"]
        tables.append({"header": column, "rows": rows})

    answers = examples["summary"]

    # IMPORTANT: we cannot pass by answers during evaluation, answers passed during training are used to
    # truncate large tables in the train set!
    
    if is_training:
        inputs = [TABLE_PROCESSOR.process_input(table, query, answer).lower() for table, query, answer in zip(tables, queries, answers)]
    else:
        inputs = [TABLE_PROCESSOR.process_input(table, query, []).lower() for table, query in zip(tables, queries)]
    labels = [TABLE_PROCESSOR.process_output(answer).lower() for answer in answers]

    model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
    labels = tokenizer(labels, max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs