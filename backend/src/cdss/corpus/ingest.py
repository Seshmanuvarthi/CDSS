"""CLI script to ingest medical corpus into local Qdrant vector database."""

import argparse
import logging
from rich.console import Console
from rich.table import Table

from ..corpus.symptom_disease import load_symptom_disease_corpus
from ..corpus.pubmed import PubMedCorpusFetcher
from ..retrieval.index import QdrantIndexManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
console = Console()


def run_ingestion(max_pubmed_docs: int = 100):
    console.print("[bold cyan]=== Starting Biomedical Corpus Ingestion ===[/bold cyan]\n")

    # Step 1: Ingest Symptom-Disease Dataset
    console.print("[bold yellow]Step 1/3: Ingesting dux-tecblic/symptom-disease-dataset...[/bold yellow]")
    symptom_docs = load_symptom_disease_corpus()
    console.print(f"Loaded [green]{len(symptom_docs)}[/green] symptom-disease records.\n")

    # Step 2: Fetch PubMed Central Literature
    console.print(f"[bold yellow]Step 2/3: Fetching {max_pubmed_docs} PMC diagnostic articles via NCBI Entrez...[/bold yellow]")
    fetcher = PubMedCorpusFetcher()
    pmc_ids = fetcher.search_diagnostic_pmc_ids(max_results=max_pubmed_docs)

    pubmed_docs = []
    if pmc_ids:
        pubmed_docs = fetcher.fetch_article_texts(pmc_ids=pmc_ids[:max_pubmed_docs], batch_size=20)
        console.print(f"Fetched [green]{len(pubmed_docs)}[/green] full-text PMC articles.\n")
    else:
        console.print("[yellow]NCBI query returned 0 IDs or rate limited; proceeding with structured clinical corpus.[/yellow]\n")

    all_docs = symptom_docs + pubmed_docs

    # Step 3: Chunk and Index into Qdrant
    console.print(f"[bold yellow]Step 3/3: Chunking ({len(all_docs)} docs) & Embedding with BioBERT into Qdrant...[/bold yellow]")
    index_mgr = QdrantIndexManager()
    total_indexed = index_mgr.chunk_and_index_documents(all_docs, batch_size=32)

    # Summary Table
    table = Table(title="Corpus Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Raw Documents", str(len(all_docs)))
    table.add_row("Symptom-Disease Pairs", str(len(symptom_docs)))
    table.add_row("PMC Articles", str(len(pubmed_docs)))
    table.add_row("Indexed Vector Points in Qdrant", str(total_indexed))
    table.add_row("Vector Dimension", "768 (BioBERT)")
    table.add_row("Storage Location", str(index_mgr.path))

    console.print(table)
    console.print("\n[bold green]✓ Ingestion and Vector Indexing Complete![/bold green]\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest clinical corpus into Qdrant")
    parser.add_argument("--max-pubmed", type=int, default=50, help="Number of PMC papers to fetch")
    args = parser.parse_args()
    run_ingestion(max_pubmed_docs=args.max_pubmed)

