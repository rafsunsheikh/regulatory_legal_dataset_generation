# Legal PDF to Instruction Dataset Pipeline

Convert legal regulation PDFs into instruction datasets suitable for fine-tuning LLMs using Ollama models.


## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.8+**
- **Ollama** installed and running ([Install Ollama](https://ollama.ai))

### 2. Install Ollama Model

```bash
ollama serve # Start the Ollama server
ollama pull llama3.1:8b
ollama pull mistral 
ollama pull phi3:mini 
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


## 🖥️ Run the Web UI

# Start the web server
uvicorn server:app --reload
```

Open http://127.0.0.1:8000 to access the UI. Uploaded PDFs are stored in `data/raw/`, processed in the background, and new instruction pairs are appended to `data/processed/dataset.jsonl`. The UI shows per-file progress, estimated time remaining, generation counts, and lets you pick the Ollama model and device per upload (edit the `MODEL_OPTIONS` array in `web/index.html` to adjust choices); task details are also exposed at `/tasks`. You can view/edit the prompt template and override run-time config (chunk size/overlap, temperature, retries) before running an upload; the edited values apply to that run only.


View the dataset and stats at http://127.0.0.1:8000/dataset. That page streams entries from `data/processed/dataset.jsonl` (paged) and shows quick stats by source/model/device and average instruction length.

You can also choose a device hint (auto/GPU/CPU) and see basic resource usage (CPU %, RSS) for each running task. The CPU-only option forces `num_gpu=0` in the Ollama call to stay off the GPU.


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

## 🎯 Next Steps

After generating your dataset:
1. Review the quality of generated instructions
2. Filter or post-process entries as needed
3. Use the dataset for fine-tuning your LLM
4. Iterate on the prompt template for better results

## 📄 License

MIT License - Feel free to use and modify for your needs.
