<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]

<br />
<div align="center">
  <h1 align="center">Legal PDF to Instruction Dataset Pipeline</h1>
  <p align="center">
    Turn legal and regulatory PDFs into instruction and response pairs for fine-tuning large language models, using a local Ollama model.
    <br /><br />
    <a href="#getting-started"><strong>Get started »</strong></a>
    <br /><br />
    <a href="#usage">Usage</a>
    &middot;
    <a href="https://github.com/rafsunsheikh/regulatory_legal_dataset_generation/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/rafsunsheikh/regulatory_legal_dataset_generation/issues">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#how-it-works">How It Works</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
      <ul>
        <li><a href="#option-1-web-ui">Option 1: Web UI</a></li>
        <li><a href="#option-2-command-line">Option 2: Command Line</a></li>
      </ul>
    </li>
    <li><a href="#configuration">Configuration</a></li>
    <li><a href="#output-format">Output Format</a></li>
    <li><a href="#troubleshooting">Troubleshooting</a></li>
    <li><a href="#next-steps">Next Steps</a></li>
    <li><a href="#project-structure">Project Structure</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project

Fine-tuning a language model on law or regulation needs many good question and answer examples, and writing them by hand is slow. This project builds those examples automatically. You give it a regulation as a PDF, and it produces a JSONL file where each line is one instruction, such as "What must an AML/CTF program include to comply with the legislation?", together with an answer grounded in the source text.

Everything runs on your own machine through [Ollama][ollama-url], so the documents never leave your computer. The repository ships with Australian Anti-Money Laundering and Counter-Terrorism Financing (AML/CTF) material in `data/raw/` as sample input.

### How It Works

1. **Extract.** `pypdf` reads the text from every page of the PDF.
2. **Clean.** Extra whitespace and empty lines are removed.
3. **Chunk.** LangChain's `RecursiveCharacterTextSplitter` splits the text into overlapping pieces (1000 characters with a 200 character overlap by default). Very small chunks under 100 characters are dropped as noise.
4. **Generate.** Each chunk is sent to an Ollama model with a prompt that asks for one instruction, an optional input and an answer, returned as JSON. Failed or badly formed replies are retried.
5. **Save.** Valid pairs are appended to `data/processed/dataset.jsonl`, so earlier runs are kept.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![Python][python-shield]][python-url]
* [![FastAPI][fastapi-shield]][fastapi-url]
* [![Ollama][ollama-shield]][ollama-url]
* [![LangChain][langchain-shield]][langchain-url]
* `pypdf`, `jsonlines`, `uvicorn`

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

* **Python 3.8 or newer**
* **Ollama**, installed and running ([install Ollama][ollama-url])

Start the Ollama server and pull the default model, `llama3.1:8b`:

```sh
ollama serve            # start the Ollama server
ollama pull llama3.1:8b # in another terminal
```

### Installation

1. Clone the repository.
   ```sh
   git clone https://github.com/rafsunsheikh/regulatory_legal_dataset_generation.git
   cd regulatory_legal_dataset_generation
   ```
2. Install the Python dependencies.
   ```sh
   pip install -r requirements.txt
   ```
3. Check that Python can reach Ollama and the model.
   ```sh
   python test_setup.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

### Option 1: Web UI

Start the web server:

```sh
uvicorn server:app --reload
```

Open http://127.0.0.1:8000, choose one or more PDF files and click **Upload & Process**. Each upload is saved in `data/raw/` and processed in the background (up to three files at a time). The page refreshes every two seconds and shows, for each file, its status, a progress bar, the current chunk, an estimated time remaining and how many pairs were generated. New pairs are appended to `data/processed/dataset.jsonl`.

The server also exposes a small JSON API:

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/upload` | Upload one or more PDF files (form field `files`) |
| `GET` | `/tasks` | List all processing tasks and their progress |
| `GET` | `/tasks/{task_id}` | Get the progress of one task |
| `GET` | `/health` | Simple health check |

### Option 2: Command Line

Put your PDF files in `data/raw/` and run:

```sh
python main.py
```

The script processes every `.pdf` file in `data/raw/`, appends the results to `data/processed/dataset.jsonl` and prints one sample entry. Note that it processes all PDFs in the folder, including the sample files that come with the repository.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Configuration

All settings live in `config.py`:

| Setting | Default | Meaning |
|---------|---------|---------|
| `OLLAMA_MODEL` | `llama3.1:8b` | The Ollama model used to write the pairs |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Characters shared between neighbouring chunks |
| `TEMPERATURE` | `0.7` | Randomness of the model (0.0 is fixed, 1.0 is creative) |
| `MAX_RETRIES` | `3` | Retries when a call fails or returns bad JSON |
| `INSTRUCTION_PROMPT` | see file | The prompt template sent with each chunk |

To use a different model, pull it with `ollama pull <model>` and change `OLLAMA_MODEL`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Output Format

The dataset is written in JSONL format, one JSON object per line:

```json
{
  "instruction": "What are the requirements for...",
  "input": "",
  "output": "According to the regulation...",
  "source_file": "regulation.pdf",
  "chunk_index": 0
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Troubleshooting

**Ollama connection failed**
* Make sure Ollama is running: `ollama serve`
* Make sure the model is installed: `ollama list`

**No PDFs found**
* Check that the PDF files are in the `data/raw/` folder
* Make sure the files end in `.pdf`

**Generation errors**
* Check the Ollama logs for model problems
* Lower `CHUNK_SIZE` if chunks are too large
* Adjust `TEMPERATURE` for better results

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Next Steps

After you generate a dataset:

1. Review the quality of the generated instructions.
2. Filter or post-process entries as needed.
3. Use the dataset to fine-tune your language model.
4. Improve the prompt template and run again for better results.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Project Structure

```
regulatory_legal_dataset_generation/
├── config.py          # model, chunking, path and prompt settings
├── main.py            # command line pipeline
├── server.py          # FastAPI server for the web UI and API
├── test_setup.py      # checks the Ollama connection
├── requirements.txt
├── src/
│   ├── extractor.py   # PDF text extraction and cleaning
│   ├── chunker.py     # text chunking
│   ├── generator.py   # Ollama calls and JSON parsing
│   └── pipeline.py    # ties the steps together and writes JSONL
├── web/index.html     # upload and progress page
└── data/
    ├── raw/           # input PDFs
    └── processed/     # dataset.jsonl output
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

The author has released this project under the MIT License. A `LICENSE` file has not been added to the repository yet.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

MD Rafsun Sheikh, [@rafsunsheikh][profile-url] on GitHub

Project link: [https://github.com/rafsunsheikh/regulatory_legal_dataset_generation][repo-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/rafsunsheikh/regulatory_legal_dataset_generation.svg?style=for-the-badge
[contributors-url]: https://github.com/rafsunsheikh/regulatory_legal_dataset_generation/graphs/contributors
[stars-shield]: https://img.shields.io/github/stars/rafsunsheikh/regulatory_legal_dataset_generation.svg?style=for-the-badge
[stars-url]: https://github.com/rafsunsheikh/regulatory_legal_dataset_generation/stargazers
[issues-shield]: https://img.shields.io/github/issues/rafsunsheikh/regulatory_legal_dataset_generation.svg?style=for-the-badge
[issues-url]: https://github.com/rafsunsheikh/regulatory_legal_dataset_generation/issues
[repo-url]: https://github.com/rafsunsheikh/regulatory_legal_dataset_generation
[profile-url]: https://github.com/rafsunsheikh
[python-shield]: https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white
[python-url]: https://www.python.org/
[fastapi-shield]: https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white
[fastapi-url]: https://fastapi.tiangolo.com/
[ollama-shield]: https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white
[ollama-url]: https://ollama.com
[langchain-shield]: https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white
[langchain-url]: https://www.langchain.com/
