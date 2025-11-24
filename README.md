# Legal PDF to Instruction Dataset Pipeline

Convert legal regulation PDFs into instruction datasets suitable for fine-tuning LLMs using Ollama's **llama3.1:8b** model.

## 📁 Project Structure

```
.
├── data/
│   ├── raw/              # Place your PDF files here
│   └── processed/        # Generated datasets saved here
├── src/
│   ├── extractor.py      # PDF text extraction
│   ├── chunker.py        # Text chunking logic
│   └── generator.py      # Instruction generation with Ollama
├── config.py             # Configuration settings
├── main.py               # Main pipeline script
└── requirements.txt      # Python dependencies
```

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Ollama** installed and running ([Install Ollama](https://ollama.ai))

### 2. Install Ollama Model

```bash
ollama pull llama3.1:8b
ollama pull mistral 
ollama pull phi3:mini 


```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your PDFs

Place your legal regulation PDF files in the `data/raw/` directory:

```bash
cp your-legal-document.pdf data/raw/
```

### 5. Run the Pipeline

```bash
python main.py
```

The pipeline will:
1. Extract text from all PDFs in `data/raw/`
2. Split text into contextual chunks
3. Generate instruction-response pairs using llama3.1:8b
4. Append results to `data/processed/dataset.jsonl`

## 🖥️ Run the Web UI

Upload multiple PDFs at once and monitor progress via a simple web interface.

```bash
# Install dependencies (includes FastAPI + uvicorn)
pip install -r requirements.txt

# Start the web server
uvicorn server:app --reload
```

Open http://127.0.0.1:8000 to access the UI. Uploaded PDFs are stored in `data/raw/`, processed in the background, and new instruction pairs are appended to `data/processed/dataset.jsonl`. The UI shows per-file progress, estimated time remaining, generation counts, and lets you pick the Ollama model per upload (edit the `MODEL_OPTIONS` array in `web/index.html` to adjust choices); task details are also exposed at `/tasks`.

You can also choose a device hint (auto/GPU/CPU) and see basic resource usage (CPU %, RSS) for each running task. The CPU-only option forces `num_gpu=0` in the Ollama call to stay off the GPU.

## ⚙️ Configuration

Edit `config.py` to customize:

- **Model**: Change `OLLAMA_MODEL` to use different models
- **Chunk Size**: Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP`
- **Temperature**: Control generation creativity (0.0-1.0)
- **Prompt Template**: Customize `INSTRUCTION_PROMPT`

## 📊 Output Format

The generated dataset is in JSONL format with the following structure:

```json
{
  "instruction": "What are the requirements for...",
  "input": "",
  "output": "According to the regulation...",
  "source_file": "regulation.pdf",
  "chunk_index": 0
}
```

## 🔧 Troubleshooting

### Ollama Connection Failed
- Ensure Ollama is running: `ollama serve`
- Verify model is installed: `ollama list`

### No PDFs Found
- Check that PDF files are in `data/raw/` directory
- Ensure files have `.pdf` extension

### Generation Errors
- Check Ollama logs for model issues
- Reduce `CHUNK_SIZE` if chunks are too large
- Adjust `TEMPERATURE` for better results

## 📝 Example Usage

```bash
# Process PDFs and generate dataset
python main.py

# View the generated dataset
head -n 1 data/processed/dataset.jsonl | python -m json.tool
```

## 🎯 Next Steps

After generating your dataset:
1. Review the quality of generated instructions
2. Filter or post-process entries as needed
3. Use the dataset for fine-tuning your LLM
4. Iterate on the prompt template for better results

## 📄 License

MIT License - Feel free to use and modify for your needs.
