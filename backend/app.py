import os
import json
import re
from typing import Optional, Dict, Any, List, Literal, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from dotenv import load_dotenv
import tldextract

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from urllib.parse import urlparse, parse_qs


# ======================
# ENV
# ======================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")


# ======================
# FASTAPI APP + TEMPLATES
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

app = FastAPI(title="PhishGuard AI – LangChain + Groq")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================
# GENAI OUTPUT SCHEMA (STRICT)
# ======================
class GenAIResult(BaseModel):
    genai_score: int
    verdict: Literal["SAFE", "SUSPICIOUS", "PHISHING"]
    top_reasons: List[str]
    notes: str


# ======================
# LANGCHAIN GROQ LLM (STRUCTURED)
# ======================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=GROQ_MODEL,
    temperature=0,
).with_structured_output(GenAIResult)


# ======================
# API SCHEMAS
# ======================
class ScanRequest(BaseModel):
    url: str
    page_title: Optional[str] = None
    page_text_snippet: Optional[str] = None
    brand_claimed: Optional[str] = None
    user_context: Optional[str] = None


class ScanResponse(BaseModel):
    url: str
    verdict: str
    risk_score: float
    ml_score: float
    genai_score: float
    reasons: List[str]
    signals: Dict[str, Any]
    genai_summary: Dict[str, Any]


# ======================
# CONFIG / HEURISTICS
# ======================
SUSPICIOUS_TLDS = {"xyz", "zip", "click", "top", "tk", "ml", "ga", "cf"}

SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl",
    "ow.ly", "is.gd", "buff.ly", "rebrand.ly"
}

SUSPICIOUS_KEYWORDS = {
    "login", "verify", "update", "secure",
    "account", "signin", "bank", "payment"
}

TRUSTED_DOMAINS = {
    "google.com",
    "chatgpt.com",
    "openai.com",
    "microsoft.com",
    "amazon.com",
    "amazon.in",
    "apple.com",
    "github.com",
    "linkedin.com",
    "facebook.com",
    "twitter.com",
}


# ======================
# UTILS
# ======================
def refang_url(url: str) -> str:
    url = str(url).strip()
    url = url.replace("hxxp://", "http://")
    url = url.replace("hxxps://", "https://")
    url = url.replace("[.]", ".")
    return url


def is_ip(host: str) -> bool:
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))


def is_trusted_domain(url: str) -> bool:
    ext = tldextract.extract(url)
    full_domain = f"{ext.domain}.{ext.suffix}".lower()
    return full_domain in TRUSTED_DOMAINS


def extract_features(url: str) -> Dict[str, Any]:
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    host = ext.fqdn.lower()
    domain = ext.domain.lower()
    query_params = parse_qs(parsed.query)

    return {
        "host": host,
        "url_length": len(url),
        "num_dots": host.count("."),
        "num_hyphens": host.count("-"),
        "has_https": url.startswith("https"),
        "looks_like_ip": is_ip(host),
        "suspicious_tld": ext.suffix.lower() in SUSPICIOUS_TLDS,

        # extra signals
        "has_at_symbol": "@" in url,
        "has_double_slash_redirect": url.count("//") > 1,
        "is_shortened_url": host in SHORTENERS,
        "has_port_in_url": parsed.port is not None,
        "https_token_in_domain": "https" in domain,
        "subdomain_count": len(ext.subdomain.split(".")) if ext.subdomain else 0,
        "query_param_count": len(query_params),
        "suspicious_keywords_in_url": any(kw in url.lower() for kw in SUSPICIOUS_KEYWORDS),
        "tld_length": len(ext.suffix),
        "numeric_domain_ratio": sum(c.isdigit() for c in domain) / max(len(domain), 1),
    }


def ml_score_calc(features: Dict[str, Any]) -> Tuple[float, List[str]]:
    score = 0
    reasons: List[str] = []

    if features["looks_like_ip"]:
        score += 20
        reasons.append("IP address used instead of domain")

    if features["suspicious_tld"]:
        score += 15
        reasons.append("Suspicious TLD detected")

    if features["url_length"] > 70:
        score += 10
        reasons.append("Unusually long URL")

    if features["num_hyphens"] >= 3:
        score += 10
        reasons.append("Too many hyphens in domain")

    if not features["has_https"]:
        score += 10
        reasons.append("HTTPS not used")

    if features["has_at_symbol"]:
        score += 10
        reasons.append("URL contains @ symbol")

    if features["has_double_slash_redirect"]:
        score += 8
        reasons.append("Multiple // redirects detected")

    if features["is_shortened_url"]:
        score += 15
        reasons.append("URL shortening service used")

    if features["has_port_in_url"]:
        score += 8
        reasons.append("Non-standard port in URL")

    if features["https_token_in_domain"]:
        score += 8
        reasons.append("Misleading 'https' in domain name")

    if features["subdomain_count"] >= 3:
        score += 10
        reasons.append("Too many subdomains")

    if features["suspicious_keywords_in_url"]:
        score += 10
        reasons.append("Suspicious keywords found in URL")

    if features["numeric_domain_ratio"] > 0.3:
        score += 8
        reasons.append("Domain contains excessive numbers")

    return min(score, 100), reasons


# ======================
# GENAI ANALYSIS (STRUCTURED)
# ======================
GENAI_SYSTEM_PROMPT = """
You are a cybersecurity expert.

IMPORTANT RULES:
- Do NOT mark a URL suspicious just because metadata is missing.
- Missing page title/snippet/brand/context does NOT imply phishing.
- If the domain is well-known and HTTPS is valid, default to SAFE unless strong phishing indicators exist.
- Be conservative: only mark PHISHING when clear signals exist.

Return structured output only.
"""


def genai_analysis(req: ScanRequest, features: Dict[str, Any]) -> GenAIResult:
    messages = [
        SystemMessage(content=GENAI_SYSTEM_PROMPT),
        HumanMessage(content=json.dumps({
            "url": req.url,
            "features": features,
            "page_title": req.page_title,
            "snippet": req.page_text_snippet,
            "brand": req.brand_claimed,
            "context": req.user_context,
        })),
    ]

    return llm.invoke(messages)


# ======================
# ROUTES
# ======================
@app.get("/", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.post("/api/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    # 1) Normalize
    real_url = refang_url(req.url)

    # 2) Features + ML score
    features = extract_features(real_url)
    ml_score, ml_reasons = ml_score_calc(features)

    # 3) Trusted domain check
    trusted = is_trusted_domain(real_url)
    if trusted:
        ml_score = min(ml_score, 10)
        ml_reasons = ["Trusted and well-known domain"]

    # 4) GenAI (on real_url)
    genai = genai_analysis(
        ScanRequest(
            url=real_url,
            page_title=req.page_title,
            page_text_snippet=req.page_text_snippet,
            brand_claimed=req.brand_claimed,
            user_context=req.user_context,
        ),
        features,
    )

    # 5) Final score (weights)
    final_score = 0.45 * ml_score + 0.55 * float(genai["genai_score"])

    # score-based verdict
    if final_score >= 75:
        verdict = "PHISHING"
    elif final_score >= 50:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SAFE"

    # 6) Override (ONLY if NOT trusted)
    if not trusted:
        if genai["verdict"] == "PHISHING":
            verdict = "PHISHING"
        elif genai["verdict"] == "SUSPICIOUS" and verdict == "SAFE":
            verdict = "SUSPICIOUS"

    # 7) Response
    return ScanResponse(
        url=real_url,
        verdict=verdict,
        risk_score=round(final_score, 2),
        ml_score=ml_score,
        genai_score=float(genai["genai_score"]),
        reasons=(ml_reasons + genai["top_reasons"])[:6],
        signals={"features": features, "trusted_domain": trusted},
        genai_summary=genai,
    )