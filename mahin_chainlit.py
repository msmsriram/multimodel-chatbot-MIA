from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import base64
from PIL import Image
from io import BytesIO
import chainlit as cl
import os
# conceptsintamil/tamil-llama-7b-instruct-v0.2:latest
@cl.on_chat_start
async def on_chat_start():
    
    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="mistral.jpeg")
    ]
    await cl.Message(content="Hello there, I am mistral. How can I help you ?", elements=elements).send()
    model = ChatOllama(model="mistral",temperature=0.1,keep_alive=3600)
    embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')
    # vectorstore = FAISS.load_local("multiple_sclerosis_faiss_latest_text_mahin", embedding,allow_dangerous_deserialization=True)

    vectorstore = FAISS.load_local("tamil_to_english_faiss_rec", embedding,allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
#     after_rag_template = """
# Answer the following question only based on the provided context.
#                     Think step by step before providing a detailed answer.Provide only the answer and don't be expressive.Please don't start your answer with 'based on the provided context'.Please don't start your answer with 'According to the text'.
#                     understand the whole given context and then provide the appropriate answer for the question
#                     <context>{context}</context>
#                     Question:{input}
# """
    after_rag_template=""" 
        Answer the following question only based on the provided context.
                        Think step by step before providing a detailed answer.Provide only the answer and don't be expressive.Please don't start your answer with 'based on the provided context'.Please don't start your answer with 'According to the text'.
                        understand the whole given context and then provide the appropriate answer for the question
                        <context>{context}</context>
                        Question:{question}
                        You should only respond to the questions only from the given context.If you are asked about anything other than given context respond by saying 'Sorry I can't help you with that.Please ask anything about Multiple Sclerosis'.
                        dont generate anything please.
        """
    # after_rag_template=""" provide me the exact answer only from the given context for the given question and do not give your own answer please.
    #     <context>{context}</context>
    #     Question:{question}
    #     """
    

    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    after_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | after_rag_prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("after_rag_chain", after_rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    after_rag_chain = cl.user_session.get("after_rag_chain")  # type: Runnable
    
    # relevant_docs = db.similarity_search('{question}')
    # # context = ""
    # relevant_images = []
    # for d in relevant_docs:
    #     # if d.metadata['type'] == 'text':
    #     #     context += '[text]' + d.metadata['original_content']
    #     # elif d.metadata['type'] == 'table':
    #     #     context += '[table]' + d.metadata['original_content']
    #     if d.metadata['type'] == 'image':
    #         # context += '[image]' + d.page_content
    #         relevant_images.append(d.metadata['original_content'])
    
    
    msg = cl.Message(content="")

    async for chunk in after_rag_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
    # os.remove(path)
   
