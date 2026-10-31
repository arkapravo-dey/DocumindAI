# Chat with Multiple PDFs — Groq Edition

A Streamlit PDF question-answering app using **Groq for chat generation**, **local Sentence Transformers embeddings**, and **FAISS** for vector search. No OpenAI API key or paid embedding API is used.

## Features

- Upload and index multiple text-based PDFs.
- Extract page-level text and retain source filename/page metadata.
- Local embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- FAISS cosine-similarity retrieval (normalized vectors + inner product).
- Groq chat completions with recent conversation history.
- Answers are instructed to stay grounded in retrieved passages and cite `[Source N]`.
- Streamlit chat interface, source list, and clear/reset controls.

## Requirements

- Windows 10/11 (or Linux/macOS)
- Python 3.11, 64-bit
- Internet access for installing packages, downloading the embedding model the first time, and calling Groq
- A Groq API key: https://console.groq.com/keys

Groq's free tier is subject to account/model rate limits and can change. It is not unlimited free usage. Local embedding generation does not use Groq tokens or an OpenAI API key.

## Windows setup (PowerShell or VS Code terminal)

1. Extract the ZIP and open a terminal in the `ask-multiple-pdfs-groq` folder.
2. Create and activate a fresh environment:

   ```powershell
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use CMD instead:

   ```bat
   .venv\Scripts\activate.bat
   ```

3. Upgrade pip and install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your Groq key:

   ```env
   GROQ_API_KEY=gsk_your_real_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   ```

   Do not commit or share `.env`.

5. Test Groq connectivity:

   ```powershell
   python test_groq.py
   ```

6. Launch the app:

   ```powershell
   python -m streamlit run app.py
   ```

   Open the local URL printed by Streamlit (usually http://localhost:8501).

## How to use

1. Upload one or more PDFs in the sidebar.
2. Select **Process PDFs**. The first run downloads the embedding model (roughly a few hundred MB); later runs use the local model cache.
3. Ask questions in the chat box. Answers are based on the top matching PDF chunks, and the UI lists the source file and page numbers.
4. Select **Clear documents and chat** to remove the current in-memory index and conversation.

## Troubleshooting

- **`GROQ_API_KEY is missing`**: ensure `.env` is in the same folder as `app.py`, named exactly `.env`, and contains `GROQ_API_KEY=...`. Restart Streamlit after editing it.
- **Groq connection error**: check whether your network/firewall permits HTTPS access to `api.groq.com` on port 443. Test using a mobile hotspot if you suspect a campus LAN restriction. The Groq API still requires internet access.
- **401 authentication error**: create/check the key at https://console.groq.com/keys.
- **429 rate limit**: wait for the account/model limit to reset or check the current limits in your Groq console. Free tier has limits.
- **Model not found/decommissioned**: check Groq's current model catalogue and update `GROQ_MODEL` in `.env` to a model available to your account.
- **No text extracted**: this app currently handles text-based PDFs. Scanned/image-only PDFs need OCR (not included).
- **First indexing is slow**: the sentence-transformer model downloads on first use; subsequent launches use the cached copy.

## Notes

- All document text and embeddings are processed locally on your computer. Only the user's question and retrieved text snippets are sent to Groq to generate an answer.
- The FAISS index is kept in Streamlit session memory and is rebuilt when PDFs are processed; this starter app does not persist indexes between restarts.
- Avoid uploading confidential documents unless permitted by your institution and the document owner. Retrieved snippets are sent to Groq for inference.
