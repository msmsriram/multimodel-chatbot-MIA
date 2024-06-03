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
from gtts import gTTS
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import os
# Set the environment variable to disable GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# conceptsintamil/tamil-llama-7b-instruct-v0.2:latest
@cl.on_chat_start
async def on_chat_start():
    
    # Sending an image with the local file path
    # elements = [
    # cl.Image(name="image1", display="inline", path="mistral.jpeg")
    # ]
    await cl.Message(content="Hello there, I am Mia (MS Information Assistant). How can I help you ?").send()
    model = ChatOllama(model="phi3:latest",temperature=0,num_gpu=0)
    set_llm_cache(InMemoryCache)
    embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')
    # vectorstore = FAISS.load_local("multiple_sclerosis_faiss_latest_text_mahin", embedding,allow_dangerous_deserialization=True)

    vectorstore = FAISS.load_local("mahin_ppt_rec", embedding,allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
#     after_rag_template = """
# You are a data scientist and an expert in analyzing images and tables and text.
# Answer the question based only on the following context, which can include text, images and tables:
# {context}
# Question: {question}
# Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
# Just return the helpful answer in as much as detailed possible.
# Answer:
# """
    after_rag_template=""" 
        Answer the following question only based on the provided context.
                        Think step by step before providing a detailed answer.Provide only the answer and don't be expressive.Please don't start your answer with 'based on the provided context'.Please don't start your answer with 'According to the text'.
                        understand the whole given context and then provide the appropriate answer for the question
                        <context>{context}</context>
                        Question:{question}
                        
        """
    # after_rag_template="""
    #     Please respond to the question using the context provided. Your answer should be direct, concise, and strictly focused on the question asked.If the question is not relevant to the Context provided, simply respond with 'I am a Chatbot designed to assist you with inquiries about Multiple Sclerosis. Feel free to ask any questions related to Multiple Sclerosis.' Do not mention anything about the context or the irrelevance of the question in your response.Avoid referring to the context in your response. Do not use phrases like 'based on the provided context' or 'According to the text' or 'The context does not directly answer your question'.
    #     <context>{context}</context>
    #     Question:{question}
    #    """

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
    embedding = embeddings.OllamaEmbeddings(model='nomic-embed-text')
    db = FAISS.load_local("multiple_sclerosis_ppt_faiss_images_latest_mahin", embeddings=embedding,allow_dangerous_deserialization=True)
    relevant_docs = db.similarity_search({"question": message.content})
    print(relevant_docs)
    # context = ""
    relevant_images = []
    for d in relevant_docs:
        # if d.metadata['type'] == 'text':
        #     context += '[text]' + d.metadata['original_content']
        # elif d.metadata['type'] == 'table':
        #     context += '[table]' + d.metadata['original_content']
        if d.metadata['type'] == 'image':
            # context += '[image]' + d.page_content
            relevant_images.append(d.metadata['original_content'])

    img_data = base64.b64decode(relevant_images[0])
    img = Image.open(BytesIO(img_data))
    img_rgb = img.convert('RGB')
    img_rgb.save("temp.jpg")
    path='temp.jpg'
    # print(context)
    

    elements = [
    cl.Image(name="image1", display="inline", path=path),
    
    ]
    msg = cl.Message(content="", elements=elements)
   
    llm_response=""
    
    async for chunk in after_rag_chain.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        llm_response=llm_response+chunk
        await msg.stream_token(chunk)
    print(llm_response)
    if 'I am a Chatbot designed to assist you with inquiries about Multiple Sclerosis.' not in llm_response:
        await msg.send()
    
    language = 'en'
    speech = gTTS(text=llm_response, lang=language, slow=False)
    speech.save("Responce.mp3")
    elements_audio = [
        cl.Audio(name="Responce.mp3", path="./Responce.mp3", display="inline"),
    ]
    await cl.Message(
        content="Here is an audio file",
        elements=elements_audio,
    ).send()
    
    # os.remove(path)
   
