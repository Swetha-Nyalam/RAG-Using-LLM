import lamini
import numpy as np
from lamini.api.embedding import Embedding
import jsonlines
from lamini import LlamaV2Runner
import os


lamini.api_key = ""
llm = lamini.Lamini("meta-llama/Llama-2-7b-chat-hf")


class Chunker:

    """
     the chuncker class has method which will split the given input text to the sizes of "chunk_size"
     with an overlap of "overlap" between the adjacent chunks

    """

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def get_chunks(self, text: str):
        text_splits = []
        i = 0
        for i in range(0, len(text), self.chunk_size - self.overlap):
            end = min(i + self.chunk_size, len(text))
            text_splits.append(text[i:end])
        return text_splits


class TranscriptEmbedding:

    """
    generate_text_from_transcript class will read the "sample_questions.jsonl"
    file and outputs the "combined_transcript_text" which is the concatination
    of "transcript" Keys from all the Json components
    """

    def extract_and_concatenate_transcripts(self):
        combined_transcript_text = ""
        with jsonlines.open("sample_questions.jsonl") as reader:
            for q in reader:
                if "transcript" in q:
                    combined_transcript_text += q["transcript"]
        return combined_transcript_text

    """
    generateEmbeddings class will generate the emebddings of combined transcript
    text and the query
    """

    def break_into_chunks(self):
        combined_transcript_retrieved = self.extract_and_concatenate_transcripts()
        return Chunker().get_chunks(combined_transcript_retrieved)

    def generate_chunks_embedding(self):
        ebd = Embedding()
        splitted_text_list = self.break_into_chunks()
        chunks_embedding = []
        for chunk in splitted_text_list:
            embedding = ebd.generate(chunk)
            chunks_embedding.append(embedding[0])
        return chunks_embedding


class QueryEmbedding:

    """
    LLM Rephrased Query - This function takes a query as input and utilizes an LLM
    (Large Language Model) to generate a rephrased version of the query.
    The rephrased query might be shorter or use different wording while
    preserving the original meaning.

    """

    def __init__(self, query: str):
        self.query = query

    def llm_repharsed_query(self):
        json_output_template = {
            "rephrased question ": "str"
        }
        llm_runner = LlamaV2Runner(
            system_prompt="Given the following query , repharse this query in a more concise way by using only the key words.Be very particular to inculde year or money or date and time or currency just as in the original query."
        )
        repharsedquestion_json_component = llm_runner(
            self.query, output_type=json_output_template)
        if isinstance(repharsedquestion_json_component, dict):
            for key, value in repharsedquestion_json_component.items():
                if key == "rephrased question ":
                    return value
        return ""

    """
    HyDe - Given a query and an instruction as a prompt to an LLM,
    it generates a hypothetical document/response or 'n' hypothetical documents/responses
    that answers the user question. And that hypothetical responses are used for
    """

    def hyde_llm_answer(self):
        json_output_template = {
            "Hypothetical_Answer": "str"
        }
        llm_prompt_hyde = LlamaV2Runner(
            system_prompt='''Consider a company named as "X". Give a hypothetical answer for this query from a hypothetical earning calls data. Very Importantly be as concise as possible by giving importance to words from the query. Do not provide any additional information like numbers/years.Do not add addtional comments on why you think the answer is correct. Query is :  '''
        )
        llm_hyde_answer = llm_prompt_hyde(
            self.query, output_type=json_output_template
        )
        if isinstance(llm_hyde_answer, dict):
            for key, value in llm_hyde_answer.items():
                if key == "Hypothetical_Answer":
                    return value
        return ""

    def generate_query_embedding(self):
        query_embedding = []
        ebd = Embedding()
        rephrased_query = self.llm_repharsed_query(self.query)
        embedding = ebd.generate(rephrased_query)
        query_embedding.append(embedding[0])
        return query_embedding


class PairwiseBubbleSortComparator:

    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    """
        *********************************************
        Not using the llm Relevance Ranking right now
        *********************************************
    """

    """
    This prompt asks the LLM to judge how relevant a text is to a query, giving a score from 0 (not relevant) to 1 (highly relevant)
    """

    def generate_prompt(self, text: str, prompt: str = ""):
        prompt = "This is a transcript from an earnings call of a company. Analyze the following text:"
        prompt += f"**Text:**\n{text}\n"
        prompt += "**How relevant is this text to the user given query based on keywords? The answer is a floating-point value ranging from 0.0 (inclusive) to 1.0 (inclusive).**"
        return prompt

    def get_relevance_score_from_llm(self, text: str, query: str):
        op_json = {
            "Relevance_Score": "float"
        }
        runner = LlamaV2Runner(
            system_prompt=self.generate_prompt(text, ""))
        relevant_score = runner(query, output_type=op_json)
        if isinstance(relevant_score, dict):
            for key, value in relevant_score.items():
                if key == "Relevance_Score":
                    return value
        return 0

    '''
       used when we are asking llm to give the relevance score for the text and query
    '''

    def get_chunked_transcript_ranks(self, query: str):
        llm_relevance_rankings = []
        genemb = TranscriptEmbedding(self)
        split_texts = genemb.break_into_chunks(self)
        for text in split_texts:
            self.llm_relevance_rankings.append(
                self.get_relevance_score_from_llm(text, query))

        """
            *********************************************
            Not using the llm Relevance Ranking right now
            *********************************************
        """
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    def __init__(self, NmostRelevantTexts: int):
        self.split_texts = []
        self.NmostRelevantTexts = NmostRelevantTexts

    def get_split_texts(self):
        genemb = TranscriptEmbedding()
        self.split_texts = genemb.break_into_chunks()

    def llm_reRanker(self, text1: str, text2: str, query: str):

        reOrdered_texts = []

        op = {
            "Most_Relevant": "str",
            "Least_Relevant": "str"
        }

        prompt = "Given two texts, {}, {}, could you re-order the tests in the decreasing order of their relevance to the given query"
        prompt += "Respond with json(op type), where most_relevant corresponds to the most relevant text to query, and least_relevant coresponds to the less relevant value to the query."
        prompt += "Query is :  "
        system_prompt = prompt.format(
            text1, text2)
        llm_runner = LlamaV2Runner(system_prompt=system_prompt)
        text_priority = llm_runner(query, output_type=op)
        if isinstance(text_priority, dict):
            for key, value in text_priority.items():
                reOrdered_texts.append(value)

        return reOrdered_texts

    def llm_reRank_texts(self, query: str, nthRelevantText: int):
        for i in range(len(self.split_texts)-2, nthRelevantText, -1):
            reOrdered_texts = self.llm_reRanker(
                self.split_texts[i], self.split_texts[i+1], query)
            self.split_texts[i] = reOrdered_texts[0]
            self.split_texts[i+1] = reOrdered_texts[1]
        # print("Swapped")
        print(len(self.split_texts))

    def topNTexts(self, query: str):
        self.get_split_texts()
        for i in range(0, self.NmostRelevantTexts):
            self.llm_reRank_texts(query, i)

        return self.split_texts


class RagRunner:

    def __init__(self, query: str):
        self.query = query

    # using default value as '3' for NmostRelevantTexts which represents the top N relevants texts for the query are moved to the start using bubblesort
    NmostRelevantTexts = 3

    def llm_query_output(self):

        op = {

            "query_output": "str"
        }
        topNTextsClass = PairwiseBubbleSortComparator(
            self.NmostRelevantTexts)
        priority_split_texts = topNTextsClass.topNTexts()
        mostRelevantTranscript = ""
        for i in range(0, self.NmostRelevantTexts):
            mostRelevantTranscript += priority_split_texts[i]

        prompt = "Act like a Analyst and answer the follwing question from the earnings call data set of a company, given text : {},"
        prompt += "Respond with json(op type), where query_output is the answer to the query from the text. Query is : "
        formatted_prompt = prompt.format(mostRelevantTranscript)
        llmRunner = LlamaV2Runner(system_prompt=formatted_prompt)
        answer_jsonl = llmRunner(self.query, output_type=op)
        if isinstance(answer_jsonl, dict):
            for key, value in answer_jsonl.items():
                if (key == "query_output"):
                    return value


instance = RagRunner(
    "What was the year-to-date sales growth for ADAS in Q3")
answer = instance.llm_query_output()
print(answer)
