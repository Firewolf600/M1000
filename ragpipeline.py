import ollama
import chromadb
from pypdf import PdfReader




#load pdf extract text


def pdfdata(pdfpath):
    reader = PdfReader(pdfpath)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n\n"
    return text.strip()





# make chunks of textdata
def makechunk(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks



#stores data in vector db 
def DB(chunks):
    vector_db = chromadb.PersistentClient(path="./chroma_db").get_or_create_collection(name="rag_data")
    
    # Check if data already exists before adding again
    alreadyexists = vector_db.get()
    existing_ids = set(alreadyexists["ids"]) if alreadyexists["ids"] else set()



    for i, chunk in enumerate(chunks):
        chunk_id = str(i)  
        if chunk_id not in existing_ids:  # Only add if it's not already in DB
            vector_db.add(ids=[chunk_id], documents=[chunk])




    print("\n Vector database updated (No duplicates added).")
    return vector_db


#RAG- Retrival
def retrieve_context(query, vector_db):
    results = vector_db.query(query_texts=[query], n_results=1)
    return results["documents"][0][0] if results["documents"] else "No relevant document found."



#RAG - Generation
def RAG(query, vector_db):
    context = retrieve_context(query, vector_db)

    if context == "No relevant document found.":
        return "I couldn't find relevant information in the document."

    response = ollama.chat(
        model="model1",
        #IDK HALF THE PROMPT IS BY RCOMMENDATION OF CO PILOT , THE LAST LINES ARE BY ME , IF U WANT CHANGE THE ENTIRE THING ACCORDING TO UR PROBLEMS AND FEATURES
        messages=[
            {"role": "system", "content": (
                "You are an intelligent assistant that answers questions STRICTLY using the provided document. "
                "If the document does not contain enough data, say 'The document does not contain enough information.' "
                "IF the document database provides enough context,give the exact refference in "" and then answeR the query"

                "DO NOT include phrases like 'I am a RAG system' or 'I analyze data.' Just give clear answers you stupid."
            )},
            {"role": "user", "content": f"Based ONLY on this document, answer the following:\n\nContext:\n{context}\n\nQuestion: {query}"}
        ],




        #temperature means how creative the answer will be , put a high number and watch ur LLM either discover something new
        #or it will start to halucinate 
        options={
    "temperature": 0.5,
    "num_ctx": 2048,  # Lower if running out of memory
    "max_tokens": 600
}

    )



    
    return response['message']['content']




#main
pdf_path = "book.pdf"  
pdf_text = pdfdata(pdf_path)  
chunks = makechunk(pdf_text)  
vector_db = DB(chunks)  





#loop to test rag
while True:
    user_query = input("\n Hello, I am M1! How may I assist you today? (Type 'exit' to quit): ")

    if user_query.lower() == "exit":
        print("M1 turned off.")
        break
    
    answer = RAG(user_query, vector_db)
    print(f"\n🔹 Answer: {answer}")
