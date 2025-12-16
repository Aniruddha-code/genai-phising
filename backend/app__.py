import os
import json
import re
import logging
from typing import Optional, Dict, Any, List, Literal, Tuple
from functools import lru_cache
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
import tldextract

from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage
from urllib.parse import urlparse, parse_qs


# ======================
# LOGGING SETUP
# ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ======================
# CONFIGURATION
# ======================
class Config:
    """Application configuration constants"""
    
    # Scoring weights
    ML_WEIGHT = 0.45
    GENAI_WEIGHT = 0.55
    
    # Thresholds
    PHISHING_THRESHOLD = 75
    SUSPICIOUS_THRESHOLD = 50
    TRUSTED_DOMAIN_MAX_SCORE = 10
    
    # URL limits
    MAX_URL_LENGTH = 2048
    LONG_URL_THRESHOLD = 70
    
    # Rate limiting
    RATE_LIMIT = "10/minute"
    
    # LLM timeout (seconds)
    LLM_TIMEOUT = 30


# ======================
# ENV
# ======================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000").split(",")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing in .env")


# ======================
# FASTAPI APP + RATE LIMITING
# ======================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="PhishGuard AI",
    description="Enterprise-grade phishing detection powered by AI",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - restricted to specific origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Templates
templates = Jinja2Templates(directory="templates")


# ======================
# DATA MODELS
# ======================
class GenAIResult(BaseModel):
    """GenAI structured output schema"""
    genai_score: int = Field(..., ge=0, le=100, description="Risk score from 0-100")
    verdict: Literal["SAFE", "SUSPICIOUS", "PHISHING"]
    top_reasons: List[str] = Field(..., max_items=5)
    notes: str


class ScanRequest(BaseModel):
    """Request model for URL scanning"""
    url: str = Field(..., min_length=1, max_length=Config.MAX_URL_LENGTH)
    page_title: Optional[str] = None
    page_text_snippet: Optional[str] = None
    brand_claimed: Optional[str] = None
    user_context: Optional[str] = None
    
    @validator('url')
    def validate_url(cls, v):
        """Validate and sanitize URL input"""
        v = v.strip()
        if not v:
            raise ValueError("URL cannot be empty")
        
        # Basic URL format check
        if not re.match(r'^https?://', v, re.IGNORECASE):
            # Try to add https://
            v = f'https://{v}'
        
        # Parse to ensure it's valid
        try:
            parsed = urlparse(v)
            if not parsed.netloc:
                raise ValueError("Invalid URL format")
        except Exception:
            raise ValueError("Invalid URL format")
        
        return v


class ScanResponse(BaseModel):
    """Response model for scan results"""
    url: str
    verdict: str
    risk_score: float
    ml_score: float
    genai_score: float
    reasons: List[str]
    signals: Dict[str, Any]
    genai_summary: Dict[str, Any]
    timestamp: str


# ======================
# THREAT INTELLIGENCE
# ======================
class ThreatIntel:
    """Threat intelligence data"""
    
    SUSPICIOUS_TLDS = {
        "xyz", "zip", "click", "top", "tk", "ml", "ga", "cf", 
        "pw", "cc", "loan", "work", "date", "racing", "stream"
    }
    
    SHORTENERS = {
        "bit.ly", "tinyurl.com", "t.co", "goo.gl",
        "ow.ly", "is.gd", "buff.ly", "rebrand.ly",
        "short.io", "cutt.ly", "s.id"
    }
    
    SUSPICIOUS_KEYWORDS = {
        "login", "verify", "update", "secure", "account", 
        "signin", "bank", "payment", "confirm", "suspended",
        "locked", "unusual", "activity", "billing", "wallet"
    }
    
    TRUSTED_DOMAINS = {
        "google.com", "chatgpt.com", "openai.com", "microsoft.com",
        "amazon.com", "amazon.in", "apple.com", "github.com",
        "linkedin.com", "facebook.com", "twitter.com", "instagram.com",
        "youtube.com", "wikipedia.org", "reddit.com", "stackoverflow.com"
    }
    
    HIGH_RISK_PATTERNS = [
        r'paypal.*verify',
        r'apple.*id.*suspend',
        r'microsoft.*account.*locked',
        r'amazon.*security.*alert',
        r'bank.*urgent',
    ]


# ======================
# LANGCHAIN GROQ LLM
# ======================
try:
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0,
        timeout=Config.LLM_TIMEOUT,
    ).with_structured_output(GenAIResult)
    logger.info(f"LLM initialized successfully with model: {GROQ_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize LLM: {e}")
    raise


# ======================
# UTILITY FUNCTIONS
# ======================
def refang_url(url: str) -> str:
    """Convert defanged URLs back to normal format"""
    url = str(url).strip()
    url = url.replace("hxxp://", "http://")
    url = url.replace("hxxps://", "https://")
    url = url.replace("[.]", ".")
    url = url.replace("[dot]", ".")
    return url


def is_ip_address(host: str) -> bool:
    """Check if host is an IP address"""
    return bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))


@lru_cache(maxsize=1000)
def is_trusted_domain(url: str) -> bool:
    """Check if URL belongs to a trusted domain (cached)"""
    try:
        ext = tldextract.extract(url)
        full_domain = f"{ext.domain}.{ext.suffix}".lower()
        return full_domain in ThreatIntel.TRUSTED_DOMAINS
    except Exception as e:
        logger.warning(f"Error checking trusted domain: {e}")
        return False


def check_high_risk_patterns(url: str) -> Tuple[bool, List[str]]:
    """Check for high-risk phishing patterns"""
    url_lower = url.lower()
    matched_patterns = []
    
    for pattern in ThreatIntel.HIGH_RISK_PATTERNS:
        if re.search(pattern, url_lower):
            matched_patterns.append(f"High-risk pattern detected: {pattern}")
    
    return len(matched_patterns) > 0, matched_patterns


def extract_features(url: str) -> Dict[str, Any]:
    """Extract URL features for ML analysis"""
    try:
        parsed = urlparse(url)
        ext = tldextract.extract(url)
        
        host = ext.fqdn.lower()
        domain = ext.domain.lower()
        query_params = parse_qs(parsed.query)
        
        # Check high-risk patterns
        has_high_risk, risk_patterns = check_high_risk_patterns(url)
        
        features = {
            "host": host,
            "url_length": len(url),
            "num_dots": host.count("."),
            "num_hyphens": host.count("-"),
            "has_https": url.startswith("https"),
            "looks_like_ip": is_ip_address(host),
            "suspicious_tld": ext.suffix.lower() in ThreatIntel.SUSPICIOUS_TLDS,
            "has_at_symbol": "@" in url,
            "has_double_slash_redirect": url.count("//") > 1,
            "is_shortened_url": host in ThreatIntel.SHORTENERS,
            "has_port_in_url": parsed.port is not None,
            "https_token_in_domain": "https" in domain,
            "subdomain_count": len(ext.subdomain.split(".")) if ext.subdomain else 0,
            "query_param_count": len(query_params),
            "suspicious_keywords_in_url": any(kw in url.lower() for kw in ThreatIntel.SUSPICIOUS_KEYWORDS),
            "tld_length": len(ext.suffix),
            "numeric_domain_ratio": sum(c.isdigit() for c in domain) / max(len(domain), 1),
            "has_high_risk_pattern": has_high_risk,
            "risk_pattern_details": risk_patterns,
        }
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features from URL '{url}': {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse URL: {str(e)}"
        )


def calculate_ml_score(features: Dict[str, Any]) -> Tuple[float, List[str]]:
    """Calculate ML-based risk score with reasons"""
    score = 0.0
    reasons: List[str] = []
    
    # Critical indicators
    if features["looks_like_ip"]:
        score += 25
        reasons.append("🚨 IP address used instead of domain name")
    
    if features["has_high_risk_pattern"]:
        score += 30
        reasons.extend(features["risk_pattern_details"])
    
    # High-risk indicators
    if features["suspicious_tld"]:
        score += 15
        reasons.append("⚠️ Suspicious top-level domain (TLD)")
    
    if features["is_shortened_url"]:
        score += 18
        reasons.append("⚠️ URL shortening service detected")
    
    if features["https_token_in_domain"]:
        score += 15
        reasons.append("⚠️ Misleading 'https' in domain name")
    
    # Medium-risk indicators
    if features["url_length"] > Config.LONG_URL_THRESHOLD:
        score += 10
        reasons.append("⚠️ Unusually long URL")
    
    if features["num_hyphens"] >= 3:
        score += 10
        reasons.append("⚠️ Excessive hyphens in domain")
    
    if features["has_at_symbol"]:
        score += 12
        reasons.append("⚠️ URL contains @ symbol (redirect trick)")
    
    if features["has_double_slash_redirect"]:
        score += 10
        reasons.append("⚠️ Multiple // redirects detected")
    
    if features["subdomain_count"] >= 3:
        score += 12
        reasons.append("⚠️ Excessive subdomains")
    
    if features["suspicious_keywords_in_url"]:
        score += 12
        reasons.append("⚠️ Suspicious keywords found in URL")
    
    if features["numeric_domain_ratio"] > 0.3:
        score += 10
        reasons.append("⚠️ Domain contains excessive numbers")
    
    # Low-risk indicators
    if not features["has_https"]:
        score += 8
        reasons.append("ℹ️ No HTTPS encryption")
    
    if features["has_port_in_url"]:
        score += 8
        reasons.append("ℹ️ Non-standard port in URL")
    
    return min(score, 100), reasons


# ======================
# GENAI ANALYSIS
# ======================
GENAI_SYSTEM_PROMPT = """You are an expert cybersecurity analyst specializing in phishing detection.

CRITICAL RULES:
1. Missing metadata (page title, snippet, brand) does NOT indicate phishing
2. Well-known domains with valid HTTPS are SAFE unless strong phishing signals exist
3. Be conservative - only mark PHISHING when clear malicious indicators are present
4. Consider: domain reputation, suspicious patterns, impersonation attempts, urgency tactics

Analyze the URL and context carefully. Return structured assessment."""


def genai_analysis(req: ScanRequest, features: Dict[str, Any]) -> Optional[GenAIResult]:
    """Perform GenAI-based URL analysis with error handling"""
    try:
        messages = [
            SystemMessage(content=GENAI_SYSTEM_PROMPT),
            HumanMessage(content=json.dumps({
                "url": req.url,
                "features": features,
                "page_title": req.page_title,
                "snippet": req.page_text_snippet,
                "brand": req.brand_claimed,
                "context": req.user_context,
            }, indent=2)),
        ]
        
        result = llm.invoke(messages)
        logger.info(f"GenAI analysis completed for URL: {req.url[:50]}...")
        return result
    
    except Exception as e:
        logger.error(f"GenAI analysis failed: {e}")
        # Return fallback result
        return GenAIResult(
            genai_score=50,
            verdict="SUSPICIOUS",
            top_reasons=["AI analysis unavailable - manual review recommended"],
            notes=f"Error during analysis: {str(e)}"
        )


def determine_final_verdict(
    ml_score: float, 
    genai_result: GenAIResult, 
    is_trusted: bool
) -> str:
    """Determine final verdict based on all signals"""
    
    # Calculate weighted score
    final_score = (
        Config.ML_WEIGHT * ml_score + 
        Config.GENAI_WEIGHT * genai_result["genai_score"]
    )
    
    # Trusted domain override
    if is_trusted:
        final_score = min(final_score, Config.TRUSTED_DOMAIN_MAX_SCORE)
        return "SAFE"
    
    # Score-based verdict
    if final_score >= Config.PHISHING_THRESHOLD:
        verdict = "PHISHING"
    elif final_score >= Config.SUSPICIOUS_THRESHOLD:
        verdict = "SUSPICIOUS"
    else:
        verdict = "SAFE"
    
    # GenAI override for non-trusted domains
    if genai_result["verdict"] == "PHISHING" and verdict != "PHISHING":
        logger.info(f"GenAI override: Upgrading to PHISHING")
        verdict = "PHISHING"
    elif genai_result["verdict"] == "SUSPICIOUS" and verdict == "SAFE":
        logger.info(f"GenAI override: Upgrading to SUSPICIOUS")
        verdict = "SUSPICIOUS"
    
    return verdict


# ======================
# API ROUTES
# ======================
@app.get("/", response_class=HTMLResponse)
async def ui(request: Request):
    """Serve the main UI"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving UI: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to load UI"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": GROQ_MODEL,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "2.0.0"
    }


@app.post("/api/scan", response_model=ScanResponse)
@limiter.limit(Config.RATE_LIMIT)
async def scan_url(request: Request, req: ScanRequest):
    """
    Scan a URL for phishing indicators
    
    - **url**: URL to analyze (required)
    - **page_title**: Optional page title for context
    - **page_text_snippet**: Optional page content snippet
    - **brand_claimed**: Optional brand the page claims to represent
    - **user_context**: Optional additional context
    """
    
    start_time = datetime.utcnow()
    logger.info(f"Scanning URL: {req.url[:100]}...")
    
    try:
        # 1. Normalize URL
        normalized_url = refang_url(req.url)
        
        # 2. Extract features
        features = extract_features(normalized_url)
        
        # 3. Check if trusted domain
        is_trusted = is_trusted_domain(normalized_url)
        logger.info(f"Domain trusted: {is_trusted}")
        
        # 4. Calculate ML score
        ml_score, ml_reasons = calculate_ml_score(features)
        
        # Override ML score for trusted domains
        if is_trusted:
            ml_score = min(ml_score, Config.TRUSTED_DOMAIN_MAX_SCORE)
            ml_reasons = ["✅ Verified trusted domain"]
        
        # 5. Perform GenAI analysis
        genai_result = genai_analysis(
            ScanRequest(
                url=normalized_url,
                page_title=req.page_title,
                page_text_snippet=req.page_text_snippet,
                brand_claimed=req.brand_claimed,
                user_context=req.user_context,
            ),
            features,
        )
        
        if not genai_result:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="AI analysis service temporarily unavailable"
            )
        
        # 6. Determine final verdict
        verdict = determine_final_verdict(ml_score, genai_result, is_trusted)
        
        # 7. Calculate final score
        final_score = (
            Config.ML_WEIGHT * ml_score + 
            Config.GENAI_WEIGHT * genai_result["genai_score"]
        )
        
        if is_trusted:
            final_score = min(final_score, Config.TRUSTED_DOMAIN_MAX_SCORE)
        
        # 8. Combine reasons
        all_reasons = ml_reasons + genai_result["top_reasons"]
        unique_reasons = list(dict.fromkeys(all_reasons))[:6]  # Remove duplicates, limit to 6
        
        # 9. Build response
        response = ScanResponse(
            url=normalized_url,
            verdict=verdict,
            risk_score=round(final_score, 2),
            ml_score=round(ml_score, 2),
            genai_score=round(genai_result["genai_score"], 2),
            reasons=unique_reasons,
            signals={
                "features": features,
                "trusted_domain": is_trusted,
                "ml_weight": Config.ML_WEIGHT,
                "genai_weight": Config.GENAI_WEIGHT,
            },
            genai_summary={
                "verdict": genai_result["verdict"],
                "score": genai_result["genai_score"],
                "reasons": genai_result["top_reasons"],
                "notes": genai_result["notes"],
            },
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        elapsed = datetime.now(timezone.utc).isoformat().total_seconds()
        logger.info(f"Scan completed in {elapsed:.2f}s - Verdict: {verdict} (Score: {final_score:.2f})")
        
        return response
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error during scan: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during scanning: {str(e)}"
        )


@app.get("/api/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "trusted_domains_count": len(ThreatIntel.TRUSTED_DOMAINS),
        "suspicious_tlds_count": len(ThreatIntel.SUSPICIOUS_TLDS),
        "model": GROQ_MODEL,
        "version": "2.0.0",
        "uptime": "operational"
    }


# ======================
# STARTUP / SHUTDOWN
# ======================
@app.on_event("startup")
async def startup_event():
    """Run on application startup"""
    logger.info("=" * 50)
    logger.info("PhishGuard AI Backend Starting...")
    logger.info(f"Model: {GROQ_MODEL}")
    logger.info(f"Allowed Origins: {ALLOWED_ORIGINS}")
    logger.info(f"Rate Limit: {Config.RATE_LIMIT}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown"""
    logger.info("PhishGuard AI Backend Shutting Down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )