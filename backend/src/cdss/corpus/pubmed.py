"""NCBI Entrez / PMC Open Access downloader for clinical diagnostic literature."""

import time
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import httpx
import logging

logger = logging.getLogger(__name__)

# Paper's query parameters (§II-B):
DIAGNOSTIC_QUERY = (
    '("symptom"[Title/Abstract] OR "clinical presentation"[Title/Abstract]) '
    'AND ("diagnosis"[Title/Abstract] OR "differential diagnosis"[Title/Abstract]) '
    'AND "open access"[filter] AND "humans"[MeSH Terms] '
    'NOT ("cancer biomarkers"[Title] OR "artificial intelligence"[Title] OR "vaccine"[Title])'
)


class PubMedCorpusFetcher:
    """Fetches full-text open access clinical articles from NCBI Entrez / PMC."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def __init__(self, email: str = "cdss-research@academic.edu", tool: str = "CDSS_Reproduce"):
        self.email = email
        self.tool = tool

    def search_diagnostic_pmc_ids(self, max_results: int = 1000) -> List[str]:
        """Search PMC for open access papers matching the diagnostic criteria."""
        params = {
            "db": "pmc",
            "term": DIAGNOSTIC_QUERY,
            "retmax": max_results,
            "retmode": "json",
            "sort": "pub_date",
            "tool": self.tool,
            "email": self.email,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.get(self.ESEARCH_URL, params=params)
                res.raise_for_status()
                data = res.json()
                id_list = data.get("esearchresult", {}).get("idlist", [])
                logger.info(f"Retrieved {len(id_list)} PMC IDs matching diagnostic search criteria.")
                return id_list
        except Exception as e:
            logger.error(f"Error querying NCBI Entrez: {e}")
            return []

    def fetch_article_texts(self, pmc_ids: List[str], batch_size: int = 50) -> List[Dict[str, Any]]:
        """Fetch article XML/text in batches and extract structured sections."""
        documents = []
        if not pmc_ids:
            return documents

        logger.info(f"Fetching full text for {len(pmc_ids)} PMC articles in batches of {batch_size}...")

        with httpx.Client(timeout=60.0) as client:
            for i in range(0, len(pmc_ids), batch_size):
                batch = pmc_ids[i:i + batch_size]
                params = {
                    "db": "pmc",
                    "id": ",".join(batch),
                    "retmode": "xml",
                    "tool": self.tool,
                    "email": self.email,
                }
                try:
                    res = client.get(self.EFETCH_URL, params=params)
                    if res.status_code == 200:
                        parsed = self._parse_pmc_xml(res.text)
                        documents.extend(parsed)
                        logger.info(f"Progress: {len(documents)}/{len(pmc_ids)} documents parsed.")
                    time.sleep(0.34)  # Respect NCBI rate limit (max 3 req/sec without API key)
                except Exception as e:
                    logger.warning(f"Failed batch {i}-{i+batch_size}: {e}")

        return documents

    def _parse_pmc_xml(self, xml_text: str) -> List[Dict[str, Any]]:
        """Extract title, abstract, and clinical body text from PMC XML."""
        docs = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//article"):
                pmc_id = "unknown"
                for aid in article.findall(".//article-id"):
                    if aid.get("pub-id-type") == "pmc":
                        pmc_id = f"PMC{aid.text}"

                # Title
                title_elem = article.find(".//article-title")
                title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

                # Abstract & Body
                abstract_elem = article.find(".//abstract")
                abstract = "".join(abstract_elem.itertext()).strip() if abstract_elem is not None else ""

                body_elem = article.find(".//body")
                body = "".join(body_elem.itertext()).strip() if body_elem is not None else ""

                combined_text = f"Title: {title}\n\nAbstract: {abstract}\n\nClinical Content: {body[:3500]}"
                if len(combined_text.strip()) > 150:
                    docs.append({
                        "doc_id": pmc_id,
                        "title": title,
                        "text": combined_text,
                        "source": "PubMedCentral-OpenAccess",
                        "kind": "clinical-paper",
                        "evidence_level": "systematic-review/study"
                    })
        except Exception as e:
            logger.warning(f"Error parsing XML chunk: {e}")
        return docs

