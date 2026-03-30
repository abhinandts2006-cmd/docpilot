import ollama
import typer
import pandas as pd
import importlib
from pathlib import Path
from chat import askai
from scrape import scrape_sitemap, scrape_site
from embed import embed_texts
import store

# Initialize config at startup
store.init_config()

app = typer.Typer()


def loadpdf(file_path: Path) -> list[str]:
    try:
        PdfReader = importlib.import_module("pypdf").PdfReader
    except Exception:
        print("PDF support is unavailable. Install dependencies again to include pypdf.")
        return []

    reader = PdfReader(str(file_path))
    texts: list[str] = []
    for page in reader.pages:
        page_text = (page.extract_text() or "").strip()
        if page_text:
            texts.append(page_text)
    return texts


def _load_csv_texts(file_path: Path) -> list[str]:
    df = pd.read_csv(file_path, sep=None, engine="python")
    if df.empty:
        return []

    texts: list[str] = []
    for _, row in df.fillna("").iterrows():
        row_text = " | ".join(f"{col}: {row[col]}" for col in df.columns)
        if row_text.strip():
            texts.append(row_text)
    return texts


@app.command()
def ingest(
    source: str,
    max_pages: int = typer.Option(20, "--max-pages", "-p", help="Max pages to crawl for websites."),
    workers: int = typer.Option(16, "--workers", "-w", help="Concurrent workers for scraping."),
    batch_size: int = typer.Option(32, "--batch-size", "-b", min=1, help="Embedding batch size for faster ingest."),
    embed_workers: int = typer.Option(0, "--embed-workers", "-e", min=0, help="Threads for chunking before embedding (0 = auto)."),
):
    """Ingest a website, sitemap, PDF, or CSV by chunking and embedding content."""
    if source.endswith("sitemap.xml"):
        texts = scrape_sitemap(source, max_workers=workers)
    elif source.startswith("http"):
        texts = scrape_site(source, max_pages=max_pages, max_workers=workers)
    else:
        source_path = Path(source).expanduser().resolve()
        if not source_path.exists() or not source_path.is_file():
            print("Unsupported source")
            return

        suffix = source_path.suffix.lower()
        if suffix == ".pdf":
            texts = loadpdf(source_path)
        elif suffix == ".csv":
            texts = _load_csv_texts(source_path)
        else:
            print("Unsupported local file type. Use .pdf or .csv")
            return

        if not texts:
            print("No extractable text found in file.")
            return
        source = str(source_path)

    embed_texts(texts, source=source, batch_size=batch_size, embed_workers=embed_workers)
    print("Ingestion complete.")

@app.command()
def ask(question: str):
    """Ask a question against the ingested docs."""
    print(f"Question: {question}")
    askai(question)
    print(f"\n {question} --> Done. \n")

@app.command()
def model(action: str, model_name = typer.Argument(None, help="Name of the model to set")):
    """List available Ollama models or set a default model for Q&A or embeddings."""
    models = ollama.list()
    config = store.load_config()
    """List available models or set a default model."""
    if action == "list":
        for i in models.models:
            print(i.model)  
    elif action == "set":
        if model_name in [m.model for m in models.models]:
            config["default_model"] = model_name
            store.save_config(config)
            print(f"Model set to: {model_name}")
        else:
            print(f"Model '{model_name}' not found. Use 'list' to see available models.")
    elif action == "setembed":
        if model_name in [m.model for m in models.models]:
            config["default_embed_model"] = model_name
            store.save_config(config)
            print(f"Embed model set to: {model_name}")
        else:
            print(f"Model '{model_name}' not found. Use 'list' to see available models.")
    else:
        print("Unsupported action. Use 'list' to see available models.")


@app.command()
def speed(profile: str = typer.Argument("balanced", help="fast, balanced, or quality")):
    """Set reply speed profile by tuning retrieval and Ollama generation settings."""
    config = store.load_config()

    profiles = {
        "fast": {
            "retrieval_k": 4,
            "max_context_chars": 2200,
            "max_doc_chars": 500,
            "num_predict": 128,
            "num_ctx": 1536,
            "temperature": 0.1,
        },
        "balanced": {
            "retrieval_k": 6,
            "max_context_chars": 3500,
            "max_doc_chars": 700,
            "num_predict": 192,
            "num_ctx": 2048,
            "temperature": 0.1,
        },
        "quality": {
            "retrieval_k": 10,
            "max_context_chars": 5500,
            "max_doc_chars": 1000,
            "num_predict": 320,
            "num_ctx": 4096,
            "temperature": 0.2,
        },
    }

    selected = profiles.get(profile.lower())
    if selected is None:
        print("Unsupported profile. Use: fast, balanced, or quality")
        return

    config.update(selected)
    store.save_config(config)
    print(f"Speed profile set to: {profile.lower()}")



if __name__ == "__main__":
    app()