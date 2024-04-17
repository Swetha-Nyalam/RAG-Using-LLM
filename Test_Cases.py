import lamini
import numpy as np
from lamini.api.embedding import Embedding
from lamini import LlamaV2Runner


class unit_test_cases:

    def hyde_test_comparing_dotProduct_hydeAnswer_and_actualQuery(self):
        hypo_answer = "Based on our analysis of historical trends and industry benchmarks, we expect the semiconductor industry to grow at a rate of 8-10% in 2022. This growth is driven by several factors, including increasing demand for high-performance computing, the growth of the Internet of Things (IoT), and the ongoing development of new technologies such as artificial intelligence (AI) and 5G. However, we also expect the industry to face some challenges in 2022, including supply chain disruptions and geopolitical tensions. Overall, we believe that the semiconductor industry will continue to be a key driver of growth for the technology sector in 2022 and beyond"
        doc_retreived_answer = "The revenue from the IoT @ Home portfolio in the fourth quarter of 2020 was RMB 1.11 billion."
        transcript = "we appear to be in the second year of a multiyear growth cycle propelled by the convergence of multiple technology drivers such as 5G, IoT, AI, and autonomous vehicle, as well as secular growth related to the progression of work from home and high-performance computing. Semiconductor industry revenues are breaking out from the historical share of the global electronics market for the first time in 15 years. More recently, domestic semiconductor supply self-sufficiency is adding another layer of investment to the secular drivers. Together all of these drivers are resulting in an increased capital intensity for the semiconductor industry and higher levels of investment in fab technology and capacity. In other words, being an essential supplier to the semiconductor wafer fab equipment market and having a nearly 100% focus on the sometimes cyclical but strong growth industry is a great place to be. With that, as a backdrop of our overall outlook for industry growth, I'll now turn to our key strategies to continue to outperform industry growth and in turn deliver strong operating leverage and cash flows. I'll begin with our strategic focus on some of the strongest markets within WFE. The three key markets for our products are etch, deposition, and EUV lithography all of which are outpacing overall industry growth due to multiple technology drivers. In NAND, the industry is investing in the technology that will take them from 96 layers to 128 layers and beyond that to 256 layer devices. At each step in the process, there is more etch and deposition, you may have heard on a recent earnings call that it's mostly etch and deposition equipment that's required to continue to build these taller stacks. Similarly, with DRAM as we go from 1y to the 1z node and then to one alpha and one beta, there is more of a need for etching deposition, and we are the leading provider of fluid delivery subsystems in disease markets. In logic, the transitions to five-nanometres and three-nanometers require more complex geometries and more precise control of fluid delivery. There has also been an increase in the number of gases used for technology advancements in both logic as well as DRAM. Over the past several years in each case as these geometries become more complex this drives the need for faster etch rates, better materials selectivity, and more precise control of the processes. The key takeaway as it relates to Ichor is that these advanced technology nodes are requiring more etch and deposition intensity and especially in the case of logic and DRAM more fluid delivery content for systems. Our other key market is EUV lithography, which is growing at rates well exceeding overall industry growth. Annual system shipments are expected to continue to increase at strong double-digit growth rates for the foreseeable future and as such, we are witnessing steady increases in our EUV gas delivery sales run rate each year. In total, each of these key technology transitions across all three device types is driving increased opportunity for all three of our key markets. This is a key driver for our revenue growth outperforming the overall industry, and our increased share of WFE from 0.9% five years ago, to 1.5% in 2020, or more than a 70% increase in our share of industry spend. Our increasing share of WFE is also due to our continued market share gains and the complementary and accretive acquisition that further enabled the expansion of our product offerings and global customer footprint. Before I update you on the progress we are making in our next-generation gas panel product development program, and our other product and regional growth initiatives, I'd like to update you on our capacity plans. As I noted earlier, we are in the second year of a multi-year growth cycle with leading industry OEMs and analysts forecasting another year of growth in 2022. Given this outlook and to support the success in our new product initiatives, we are already or are actively adding capacity in our gas panel integration, machining, and well-meant businesses. On our las"
        query = "What is the expected growth rate for the semiconductor industry in 2022"
        edb = Embedding()
        hypo_answer_embedding = edb.generate(hypo_answer)
        doc_retreived_answer_embedding = edb.generate(doc_retreived_answer)
        transcript_embedding = edb.generate(transcript)
        query_embedding = edb.generate(query)
        cosine_similarity_hypo = np.dot(hypo_answer_embedding[0],
                                        transcript_embedding[0])
        cosine_similarity_query = np.dot(
            query_embedding[0], transcript_embedding[0])

        print(cosine_similarity_hypo)
        print(cosine_similarity_query)

    def llm_reRan_text_onTwoTexts(self):
        op = {
            "Most_Relevant": "str",
            "Least_Relevant": "str"
        }
        text_splits = ["Newyork is the city of lights.",
                       "Paris is a beautiful city.", "DavinCi painted MonaLisa in Paris"]
        prompt = "Given three texts, {}, {}, could you re-order the tests in the decreasing order of their relevance to the given query, Respond with json(op type), where most_relevant corresponds to the most relevant text to query, and least_relevant coresponds to the less relevant value to the query. Query is :  "
        text_prompt = prompt.format(
            text_splits[0], text_splits[2])
        print(text_prompt)
        llm_runner = LlamaV2Runner(system_prompt=text_prompt)
        text_priority = llm_runner(
            " Tell me about Paris?", output_type=op)
        return text_priority

    def text_arry_list(self):
        List = ["one", "two", "three"]
        text_List = ["1", "2", "3"]
        text_List[0:2] = List[0:2]
        print(text_List)


text = unit_test_cases()
text.text_arry_list()
