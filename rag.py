import os
from dotenv import load_dotenv
load_dotenv()
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client=chromadb.PersistentClient(path="./chroma_db")
collection=chroma_client.get_or_create_collection("papers")

def load_pdf(path: str)-> list[str]:
     reader=PdfReader(path)
     chunks=[]
     for page in reader.pages:
          text=page.extract_text()
          if text:
               chunks.append(text)
     return chunks

def embed_papers(folder: str):
     for filename in os.listdir(folder):
          if filename.endswith(".pdf"):
               path=os.path.join(folder,filename)
               chunks=load_pdf(path)
               for i,chunk in enumerate(chunks):
                    embedding=embedder.encode(chunk).tolist()
                    collection.add(
                         documents=[chunk],
                         embeddings=[embedding],
                         ids=[f"{filename}_page_{i}"]
                    )
               print(f"Embedded'{filename}")

def retrieve(question:str, n_results:int=3)-> list[str]:
     embedding=embedder.encode(question).tolist()
     results=collection.query(
          query_embeddings=[embedding],
          n_results=n_results
     )
     return results['documents'][0]


def rag_answer(question:str)->str:
      chunks=retrieve(question,n_results=3)
      context="\n\n".join(chunks)

      from groq import Groq
      groq_raw=Groq(api_key=os.getenv("GROQ_API_KEY"))
      response=groq_raw.chat.completions.create(
           model="llama-3.3-70b-versatile",
           messages=[
                {
                     "role":"system",
                     "content":("You are a sports science assistant. Answer the question using only the context provided. "
                    "Cite specific findings from the context. If the context doesn't contain the answer, say so."
                     )
                },
                {
                     "role":"user",
                     "content":f"Context:\n{context}\n\nQuestion: {question}"
                }
           ]
      )
      return response.choices[0].message.content


