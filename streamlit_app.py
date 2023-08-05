import streamlit as st
from streamlit_ace import st_ace, KEYBINDINGS, LANGUAGES, THEMES
from llama_index import SimpleDirectoryReader, LangchainEmbedding, GPTListIndex,VectorStoreIndex, PromptHelper
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LLMPredictor, ServiceContext
import PyPDF2
import torch
from langchain.llms.base import LLM
from transformers import pipeline
from typing import Mapping, Optional, Any, List
from langchain.callbacks.manager import CallbackManagerForLLMRun
import json



class CustomLLM(LLM):
    n: int
    model_name = "google/flan-t5-large"
    pipeline = pipeline("text2text-generation", model=model_name, device=0, model_kwargs={"torch_dtype":torch.bfloat16})
    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        out = self.pipeline(prompt, max_length=9999)[0]["generated_text"]
        return out[:self.n]

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

llm_predictor = LLMPredictor(llm=CustomLLM(n=500))


# Using the HF embeddings for the model
hfemb = HuggingFaceEmbeddings()
embed_model = LangchainEmbedding(hfemb)



with st.form("upload-form", clear_on_submit=True):
                uploaded_file = st.file_uploader("Choose a file", accept_multiple_files=False,
                                                 type=['pdf'],
                                                 help="Upload a file to annotate")
                submitted = st.form_submit_button("Upload")

if uploaded_file is not None :
  pdf_file = uploaded_file
#extracting text from pdf
  pdf_reader = PyPDF2.PdfReader(pdf_file)
  text = ""
  for page_num in range(len(pdf_reader.pages)):
      page = pdf_reader.pages[page_num]
      text += page.extract_text()



  from llama_index import Document

  text_list = [text]

  documents = [Document(text=t) for t in text_list]



# set number of output tokens
  num_output = 500
# set maximum input size
  max_input_size = 512
# set maximum chunk overlap
  max_chunk_overlap = 1

  prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)



# index = GPTSimpleVectorIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)

# index = GPTListIndex(documents, embed_model=embed_model, llm_predictor=llm_predictor)

# index.save_to_disk('index.json')

  service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, embed_model=embed_model)
  index = VectorStoreIndex.from_documents(documents, service_context=service_context)


# use for debugging.
  import logging

  logging.getLogger().setLevel(logging.CRITICAL)


  query_engine = index.as_query_engine()
  response = query_engine.query("What is the name of this product ?")
  response2 = query_engine.query("What is the description of this product")
  minimum = query_engine.query("Give me  the integer response of the minimum operation temperature without including the unit symbole")
  maximum = query_engine.query("What is the maximum operation temperature range?")
  unit = query_engine.query("What is the unit of operation temperature ?")



  composante_dic = {"name": 'Name', "type": "0", "value": ""}
  composante_dic['value'] = response.response

  parametre = {"name": 'Description', "type": "0", "value": ""}
  parametre['value'] = response2.response

  operation_temperature = {"name": "Operation Temperature", "type": '4', "minimum": "", "maximum": "", "unit": ""}
  operation_temperature['minimum'] = minimum.response
  operation_temperature['maximum'] = maximum.response
  operation_temperature['unit'] = unit.response

  # Interface Streamlit
  c1, c2 = st.columns([1, 4])
  c2.subheader("Parameters")
  with c1:
      name = c2.text_input("name", value=composante_dic['value'])
      description = c2.text_area("description", value=parametre['value'], max_chars=200)
      unit = c2.text_input("unit", value=operation_temperature['unit'])
      maximum = c2.text_input("maximum", value=operation_temperature['maximum'])
      minimum = c2.text_input("minimum", value=operation_temperature['minimum'])

  # Bouton d'enregistrement
  if st.button("Enregistrer"):
      # Créer un dictionnaire contenant les valeurs saisies par l'utilisateur
      data = {
          "composante_dic": {"name": name, "type": composante_dic['type'], "value": composante_dic['value']},
          "parametre": {"name": "Description", "type": parametre['type'], "value": description},
          "operation_temperature": {
              "name": "Operation Temperature",
              "type": operation_temperature['type'],
              "minimum": minimum,
              "maximum": maximum,
              "unit": unit,
          },
      }

      # Enregistrer les données dans un fichier JSON
      with open("parametres.json", "w") as json_file:
          json.dump(data, json_file)

      # Afficher un message de confirmation
      st.success("Les paramètres ont été enregistrés dans parametres.json.")








#st.download_button(label=model.download_text,
#data=file,
#file_name=annotation_selection + ".json",
#mime='application/json',
#help=model.download_hint)



