from langchain.embeddings.openai import OpenAIEmbeddings
import os
from pinecone import ServerlessSpec
#from main import *
from pinecone import Pinecone
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from openai import OpenAI
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
from pinecone import ServerlessSpec

cloud = 'aws'
region = 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'renewable1'

# get openai api key from platform.openai.com
OPENAI_API_KEY =  os.environ.get('OPENAI_API_KEY')

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    # Initialize the language model with the specified parameters.
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    # Set up the retriever with the given vector store and search parameters.
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # Create a retrieval-based QA chain that returns the source documents along with the answers.
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

    # Invoke the chain with the provided question and get the response.
    answer = chain.invoke(q)

    # Print the result from the answer.
    print(answer['result'])

    # Print reference information.
    print('Reference:\n')
    # for doc in answer["source_documents"]:
    #     raw_dict = doc.metadata
    #     print("Page number:", raw_dict['page'], "Filename:", raw_dict['source'])
    for x in range(len(answer["source_documents"][0].metadata)):
        raw_dict = answer["source_documents"][x].metadata
        print("Page number:", raw_dict['page'], "Filename:", raw_dict['source'])

    # If needed, return the answer object.
    return answer

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
import streamlit as st

# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Ministry of New and Renewable Energy (Documents)...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to Ministry of New and Renewable Energy

    [Documents Repository](https://drive.google.com/drive/folders/1lC40pdVKJJXFYIy3QNgYOTlk8yo6AEoY?usp=sharing)
    ''')
    
    # Adding the new list with green bullet points
    st.markdown('''
    <div style="color: green;">
    <ul>
        <li>Annual-Report-2021-2022.pdf</li>
        <li>Annual-Report-2022-2023.pdf</li>
        <li>Biomass-Potential-Assessment-Report (M...</li>
        <li>Broad_Overview_of_RE_Generation_Febru...</li>
        <li>CEA Optimal mix report 2029-30 FINAL.pdf</li>
        <li>ESS_Roadmap for India 2019-2032 by NITI...</li>
        <li>Global Offshore Wind Policy Assessment ...</li>
        <li>Inception Report.pdf</li>
        <li>INDIA'S ENERGY STORAGE MISSION.pdf</li>
        <li>Metocean Study.pdf</li>
        <li>Monthly_RE_Generation_report_February...</li>
        <li>National Framework for Promoting Ener...</li>
        <li>National Green Hydrogen Mission.pdf</li>
        <li>Renewable_Energy_Market_Update.pdf</li>
        <li>Report of The Technical Committee on St...</li>
        <li>Report On Supply Chain, Ports And Logis...</li>
        <li>Report-onIndiaRenewableElectricityRoad...</li>
        <li>Small Hydro Power.pdf</li>
        <li>State-wise installed capacity of Renewabl...</li>
        <li>State-wise installed capacity of Renewabl...</li>
        <li>Strategy for Establishment of offshore wi...</li>
        <li>WIND DATA SHARING POLICY.pdf</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

    # Add vertical space
    st.markdown('''
    ---

    **In case of suggestions/feedback/Contributions please reach out to:**
    [NIC Training Unit @ nictu@lbsnaa.gov.in]
    ''')


# # Sidebar contents
# with st.sidebar:
#     st.title('💬 LLM Chat App on Ministry of New and Renewable Energy (Documents)...')
#     st.markdown('''
#     ## About
#     This GPT helps in answering questions related to document of Ministry of Tribal Affairs



#     [Documents Repository](https://drive.google.com/drive/folders/1lC40pdVKJJXFYIy3QNgYOTlk8yo6AEoY?usp=sharing)
 
#     ''')
#     #add_vertical_space(5)
#     st.write('Made by LBSNAA for learning purpose](https://www.lbsnaa.gov.in/)')

# def main():
#     #st.title("Question and Answering App powered by LLM and Pinecone")

#     text_input = st.text_input("Ask your query...") 
#     if st.button("Ask Query"):
#         if len(text_input)>0:
#             #st.info("Your Query: " + text_input)
#             #answer = qa_with_sources(text_input)
#             #st.success(answer)
#             answer = ask_and_get_answer(vectorstore,text_input)
#             st.success(answer)
#             #st.success(answer['result'])
#             #st.success(answer['Reference:\n'])

# if __name__ == "__main__":
#     main()
#import streamlit as st
#from your_module import ask_and_get_answer, vectorstore  # Assuming 'vectorstore' is initialized in 'your_module.py'

def display_answer(answer):
    st.write("### Query")
    st.write(answer['query'])

    st.write("### Result")
    result = answer['result'].replace('\n', '  \n')  # Ensuring markdown line breaks
    st.markdown(result)

    if "source_documents" in answer:
        st.write("### Reference Documents")
        for i, doc in enumerate(answer["source_documents"], start=1):
            st.write(f"#### Document {i}")
            st.write(f"**Page number:** {doc.metadata['page']}")
            st.write(f"**Source file:** {doc.metadata['source']}")
            content = doc.page_content.replace('\n', '  \n')  # Ensuring markdown line breaks
            st.markdown(content)

def main():
    st.title("Question and Answering App powered by LLM and Pinecone on Ministry of New and Renewable Energy (Documents)...")
    text_input = st.text_input("Ask your query...") 

    if st.button("Ask Query"):
        if len(text_input) > 0:
            answer = ask_and_get_answer(vectorstore, text_input)
            display_answer(answer)

# The main function call
if __name__ == "__main__":
    main()
