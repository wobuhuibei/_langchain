from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ChatVectorDBChain
from chatglm_llm import ChatGLM

def init_knowledeg_vector_store(filepath):
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    hf = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs,
    )
    loader = UnstructuredFileLoader(filepath, mode="elements")
    docs = loader.load()

    faiss = FAISS.from_embeddings(docs, hf)

    return faiss

def get_knowledge_based_answer(query, vector_store, chat_history=[]):
    condese_propmt_template = """任务: 给一段对话和一个后续问题，将后续问题改写成一个独立的问题。确保问题是完整的，没有模糊的指代。
        ----------------
        聊天记录：
        {chat_history}
        ----------------
        后续问题：{question}
        ----------------
        改写后的独立、完整的问题："""

    new_question_prompt = PromptTemplate.from_template(condese_propmt_template)
    chatglm = ChatGLM()
    chatglm.history = chat_history
    knowledge_chain = ChatVectorDBChain.from_llm(
        llm=chatglm,
        vectorstore=vector_store,
        condense_question_prompt=new_question_prompt,
    )

    knowledge_chain.return_source_documents = True
    knowledge_chain.top_k_docs_for_context = 10

    result = knowledge_chain({"question": query, "chat_history": chat_history})
    return result, chatglm.history

if __name__ == '__main__':
    filepath = input("Input your local knowledge file path 请输入本地知识文件路径：")
    vector_store = init_knowledeg_vector_store(filepath)

    history = []
    while True:
        query = input("Input your question 请输入问题：")
        resp, history = get_knowledge_base_answer(query = query,
                                                  vector_store = vector_store,
                                                  chat_history = history)

        print(resp)
