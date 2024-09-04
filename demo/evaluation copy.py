from tqdm import tqdm
import pandas as pd
from extractor import default_a_extractor, default_q_extractor


class Record:
    """This is a class to save 1 record in the evaluation"""

    def __init__(
        self,
        original_question,
        loop,
        qa_method,
        question_extractor,
        answer_extractor,
    ):
        self.original_question = original_question
        self.questions_loop = [original_question]
        self.answers_loop = []
        self.qa_method = qa_method
        self.loop = loop
        self.question_extractor = question_extractor
        self.answer_extractor = answer_extractor
        self.prompt4question = None
        self.prompt4answer = None

    def question_extract(self, *args, **kwargs):
        return self.question_extractor(*args, **kwargs)

    def answer_extract(self, *args, **kwargs):
        return self.answer_extractor(*args, **kwargs)

    def qa_loop(self):
        """
        Generates a list of questions and answers based on a given initial question.

        Args:
            ask (function): A function that takes a model, tokenizer, device, and a question as input and returns an answer.
            q0 (str): The initial question.

        Returns:
            tuple: A tuple containing two lists - q_list and a_list.
                - q_list (list): A list of questions generated during the process.
                - a_list (list): A list of answers corresponding to the generated questions.
        """
        # q_prompt = "What is the most likely question for this answer:"
        for i in range(self.loop):
            old_question = self.questions_loop[i]
            answer = self.qa_method(f"{self.prompt4question}{old_question}")
            answer = self.answer_extract(answer)
            self.answers_loop.append(answer)
            new_question = self.qa_method(f"{self.prompt4answer}{answer}")
            new_question = self.question_extract(new_question)
            self.questions_loop.append(new_question)



class Evaluation:
    """
    Initializes an Evaluation object.

        model: The model to be evaluated.
        tokenizer: The tokenizer used for tokenization.
        metric: The metric used for evaluation.
        model_name: The name of the model.
        evaluation_data: The data used for evaluation.
        loop: The number of loops for evaluation.
        language: The language used for evaluation.
        task: The task being performed.
        device: The device used for evaluation.
        backup_db: The backup database.
        q_extractor: The question extractor.
        a_extractor: The answer extractor.



        ask: A function that takes a model, tokenizer, device, and a question as input and returns an answer.
        q0: The initial question.

        A tuple containing two lists - q_list and a_list.
        - q_list: A list of questions generated during the process.
        - a_list: A list of answers corresponding to the generated questions.


        ask: The query string.

        question_records: A list of questions related to the query.
        answer_records: A list of answers related to the query.


        ask: The question to evaluate.

        questions: The list of retrieved questions.
        answers: The list of retrieved answers.


        mode: The mode for which to calculate the score. Valid values are "answer" and "question".




        questions: A list of strings representing the questions.
        answers: A list of strings representing the corresponding answers.
    """

    def __init__(
        self,
        model,
        tokenizer,
        metric,
        model_name,
        evaluation_data,
        loop,
        task,
        backup_db,
        q_extractor,
        a_extractor,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.metric = metric
        self.database = backup_db
        self.evaluation_data = evaluation_data
        self.model_name = model_name
        self.loop = loop
        self.task = task
        self.scores = None
        if q_extractor is None:
            self.q_extractor = default_q_extractor
        else:
            self.q_extractor = q_extractor
        if a_extractor is None:
            self.a_extractor = default_a_extractor
        else:
            self.a_extractor = a_extractor

    def recall_qa(self, ask, q0):
        """
        Generates a list of questions and answers based on a given initial question.

        Args:
            ask (function): A function that takes a model, tokenizer, device, and a question as input and returns an answer.
            q0 (str): The initial question.

        Returns:
            tuple: A tuple containing two lists - q_list and a_list.
                - q_list (list): A list of questions generated during the process.
                - a_list (list): A list of answers corresponding to the generated questions.
        """
        q_list = []
        a_list = []
        q_list.append(q0)
        if self.language == "zh":
            q_prompt = "这个答案的问题最可能是什么:"
        elif self.language == "en":
            q_prompt = "What is the most likely question for this answer:"
        for i in range(self.loop):
            a = ask(self.model, self.tokenizer, self.device, q_list[i])
            a = self.a_extractor(a)
            a_list.append(a)
            q_next = ask(self.model, self.tokenizer, self.device, q_prompt + a)
            q_next = self.q_extractor(q_next)
            q_list.append(q_next)
        return q_list, a_list

    def recall_qas(self, ask):
        """
        Retrieves questions and answers related to a given query.

        Parameters:
        - ask (str): The query string.

        Returns:
        - question_records (list): A list of questions related to the query.
        - answer_records (list): A list of answers related to the query.
        """
        for i in tqdm(range(len(self.evaluation_data))):
            questions, answers = self.recall_qa(ask, self.evaluation_data[i])
        question0_list = [str(q) for q in self.evaluation_data if isinstance(q, str)]
        question_records = []
        answer_records = []
        for question in tqdm(question0_list):
            questions, answers = self.recall_qa(ask, question)
            question_records.append(questions)
            answer_records.append(answers)
        return question_records, answer_records

    def evalutate(self, ask):
        """
        Evaluate the given question and retrieve the corresponding questions and answers.

        Parameters:
        - ask (str): The question to evaluate.

        Returns:
        - questions (list): The list of retrieved questions.
        - answers (list): The list of retrieved answers.
        """
        self.questions, self.answers = self.recall_qas(ask)
        return self.questions, self.answers

    def get_score(self, mode="answer"):
        """
        Calculate the score for the given mode.

        Parameters:
            mode (str): The mode for which to calculate the score. Valid values are "answer" and "question".

        Raises:
            ValueError: If the mode is not "answer" or "question".

        Returns:
            None
        """
        if mode == "answer":
            records = self.answers
        elif mode == "question":
            records = self.questions
        else:
            raise ValueError("mode should be answer or question")
        for i in range(self.loop):
            predictions = [record[0] for record in records]
            references = [record[i] for record in records]
            score = self.metric.compute(predictions=predictions, references=references)
            score.update({"loop": i, "refer": "0", "mode": mode})
            self.scores.append(score)
            print(f"loop{i}:{score}")
        for i in range(1, self.loop):
            predictions = [record[i - 1] for record in records]
            references = [record[i] for record in records]
            score = self.metric.compute(predictions=predictions, references=references)
            score.update({"loop": i, "refer": "n-1", "mode": mode})
            self.scores.append(score)
            print(f"loop{i}:{score}")

    def load_qa(self, questions, answers):
        """
        Load the given questions and answers into the object.

        Args:
            questions (list): A list of strings representing the questions.
            answers (list): A list of strings representing the corresponding answers.

        Returns:
            None
        """
        self.questions = questions
        self.answers = answers

    def write_scores_to_csv(self):
        """
        Writes the scores to a CSV file.

        This method takes the scores stored in the `self.scores` attribute and writes them to a CSV file.
        The CSV file is named based on the `model_name` and `task` attributes of the object.

        Parameters:
            None

        Returns:
            None
        """
        df = pd.DataFrame(self.scores)
        df.to_csv(f"score/{self.model_name}_{self.task}_scores.csv", index=False)

    def write_qa2db(self):
        """
        Writes the QA data to the database.

        This method iterates over the range of `self.loop` and constructs a dictionary `record` with the following keys:
        - "model": The name of the model.
        - "language": The language used.
        - "question": The question at index `i` in the `self.questions` list.
        - "answer": The answer at index `i` in the `self.answers` list.
        - "loop": The value of `self.loop`.
        - "task": The task being performed.

        The constructed `record` dictionary is then inserted into the database using the `insert_one` method of `self.database`.

        Parameters:
        - None

        Returns:
        - None
        """
        for i in range(self.evaluation_data):
            rerord = {
                "model": self.model_name,
                "language": self.language,
                "question": self.questions[i],
                "answer": self.answers[i],
                "loop": self.loop,
                "task": self.task,
            }
            self.database.insert_one(rerord)
