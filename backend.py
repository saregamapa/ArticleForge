"""
ArticleForge - FastAPI backend.
OpenAI-powered article generator and helpful tools. Same API contract as frontend.
"""

import asyncio
import logging
import os
import time
import urllib.request
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, JSONResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

# Rate limiting
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SEC = 60
_request_timestamps: defaultdict = defaultdict(list)

def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW_SEC
    timestamps = _request_timestamps[client_ip]
    while timestamps and timestamps[0] < window_start:
        timestamps.pop(0)
    if len(timestamps) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again in a minute.")
    timestamps.append(now)

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"

# OpenAI Client
api_key = os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=api_key)

# Lifespan: Check for API key on startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not api_key or not api_key.startswith("sk-"):
        logger.error("OPENAI_API_KEY is not set or invalid in .env file. Please set a valid key starting with 'sk-'.")
        raise RuntimeError("OPENAI_API_KEY is required to start the application. Check your .env file.")
    logger.info("OPENAI_API_KEY validated successfully.")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend and policy pages)
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(Path(__file__).parent / "index.html")

@app.get("/{path:path}", include_in_schema=False)
async def serve_static(path: str, response: Response):
    file_path = Path(__file__).parent / path
    if file_path.is_file():
        return FileResponse(file_path)
    response.status_code = 404
    return "Not found"

# Chat helper
async def chat(system: str, user: str, max_tokens: int = 4000) -> str:
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Chat failed: %s", e)
        raise HTTPException(status_code=500, detail="Failed to generate content.")

# Rate limit wrapper
def _rate_limited(request: Request):
    ip = _client_ip(request)
    _check_rate_limit(ip)

# Models (abridged for brevity; add all from your original if needed)
class GenerateRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    audience: str = Field(default="", max_length=200)
    tone: str = Field(default="Professional", max_length=50)
    wordcount: int = Field(..., ge=100, le=2000)
    keywords: Optional[str] = Field(default=None, max_length=300)

class GenerateResponse(BaseModel):
    article: str

class TitleTaglineRequest(BaseModel):
    industry: str = Field(..., min_length=1, max_length=300)

class TitleTaglineResponse(BaseModel):
    titles: list[str]

class SeoKeywordsRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    target_audience: str = Field(default="", max_length=200)

class SeoKeywordsResponse(BaseModel):
    keywords: list[str]

class MetaDescriptionRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)
    keywords: str = Field(default="", max_length=400)

class MetaDescriptionResponse(BaseModel):
    meta_description: str

class ProductDescriptionRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=200)
    features: str = Field(..., min_length=1, max_length=2000)
    tone: str = Field(default="Professional", max_length=50)

class ProductDescriptionResponse(BaseModel):
    description: str

class FaqItem(BaseModel):
    question: str
    answer: str

class FaqRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=500)

class FaqResponse(BaseModel):
    faqs: list[FaqItem]

class SocialPostRequest(BaseModel):
    post_topic: str = Field(..., min_length=1, max_length=500)
    tone: str = Field(default="Professional", max_length=50)
    platform: str = Field(..., min_length=1, max_length=50)

class SocialPostResponse(BaseModel):
    posts: list[str]

class InstagramCaptionRequest(BaseModel):
    image_topic: str = Field(..., min_length=1, max_length=500)
    tone: str = Field(default="Friendly", max_length=50)
    emoji_style: str = Field(default="Moderate", max_length=50)

class InstagramCaptionResponse(BaseModel):
    caption: str

class YoutubeScriptRequest(BaseModel):
    video_topic: str = Field(..., min_length=1, max_length=500)
    audience: str = Field(default="", max_length=200)
    tone: str = Field(default="Engaging", max_length=50)

class YoutubeScriptResponse(BaseModel):
    script: str

class YoutubeThumbnailRequest(BaseModel):
    video_topic: str = Field(..., min_length=1, max_length=500)
    tone: str = Field(default="Engaging", max_length=50)

class YoutubeThumbnailResponse(BaseModel):
    titles: list[str]

class AiImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    style: str = Field(default="Natural", max_length=100)

class AiImageResponse(BaseModel):
    image_url: str  # Data URL (data:image/png;base64,...) or DALL-E URL of generated image

class MarketingVideoRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=200)
    features: str = Field(..., min_length=1, max_length=2000)
    tone: str = Field(default="Professional", max_length=50)

class MarketingVideoResponse(BaseModel):
    script: str

class CtaRequest(BaseModel):
    action: str = Field(..., min_length=1, max_length=200)
    product_service: str = Field(..., min_length=1, max_length=500)
    tone: str = Field(default="Professional", max_length=50)

class CtaResponse(BaseModel):
    ctas: list[str]

class EmailRequest(BaseModel):
    email_topic: str = Field(..., min_length=1, max_length=500)
    audience: str = Field(default="", max_length=200)
    tone: str = Field(default="Professional", max_length=50)

class EmailResponse(BaseModel):
    email: str

class PageSummaryRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=25000)

class PageSummaryResponse(BaseModel):
    summary: str

class CompetitorAnalysisRequest(BaseModel):
    competitor_url: str = Field(..., min_length=1, max_length=500)
    topic: str = Field(..., min_length=1, max_length=500)

class CompetitorAnalysisResponse(BaseModel):
    suggestions: list[str]

class InternalLinksRequest(BaseModel):
    article_content: str = Field(..., min_length=1, max_length=15000)
    website_pages: str = Field(..., min_length=1, max_length=5000)

class InternalLinksResponse(BaseModel):
    links: list[str]

class SitemapRequest(BaseModel):
    website_url: str = Field(..., min_length=1, max_length=500)

class SitemapResponse(BaseModel):
    sitemap: str

class LandingPageRequest(BaseModel):
    product_name: str = Field(..., min_length=1, max_length=200)
    target_audience: str = Field(default="", max_length=200)
    features: str = Field(..., min_length=1, max_length=2000)
    tone: str = Field(default="Professional", max_length=50)

class LandingPageResponse(BaseModel):
    headline: str
    subheadline: str
    features: list[str]
    cta: str

def _parse_landing(content: str) -> dict:
    headline = subheadline = cta = ""
    features = []
    in_feat = False
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("HEADLINE:"):
            headline = line.split(":", 1)[1].strip()
            in_feat = False
        elif line.upper().startswith("SUBHEADLINE:"):
            subheadline = line.split(":", 1)[1].strip()
            in_feat = False
        elif line.upper().startswith("CTA:"):
            cta = line.split(":", 1)[1].strip()
            in_feat = False
        elif line.upper() == "FEATURES:":
            in_feat = True
        elif in_feat and line:
            features.append(line.strip(".- "))
    return {"headline": headline or "Headline", "subheadline": subheadline or "", "features": features[:8] or [""], "cta": cta or "Get started"}

# Endpoint example (add others like /generate/title-tagline, etc.)
@app.post("/generate", response_model=GenerateResponse)
async def generate_article(request: Request, body: GenerateRequest):
    _rate_limited(request)
    system = "You are a professional article writer. Write a high-quality, SEO-optimized article in Markdown format."
    user = f"Topic: {body.topic}. Audience: {body.audience or 'General'}. Tone: {body.tone}. Word count: ~{body.wordcount}. Keywords: {body.keywords or 'None'}."
    content = await chat(system, user, max_tokens=body.wordcount + 200)
    return GenerateResponse(article=content)

@app.post("/generate/title-tagline", response_model=TitleTaglineResponse)
async def title_tagline(request: Request, body: TitleTaglineRequest):
    _rate_limited(request)
    user = f"Industry/Product: {body.industry}. Generate 5 catchy title and tagline pairs, one per line as 'Title - Tagline'."
    content = await chat("You are a branding expert. Output one per line, no numbering.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:5]
    return TitleTaglineResponse(titles=lines or [content[:200]])

@app.post("/generate/seo-keywords", response_model=SeoKeywordsResponse)
async def seo_keywords(request: Request, body: SeoKeywordsRequest):
    _rate_limited(request)
    user = f"Topic: {body.topic}. Audience: {body.target_audience or 'General'}. Suggest 10-15 SEO keywords/phrases, one per line."
    content = await chat("You are an SEO expert. Output one keyword per line, no numbering.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:15]
    return SeoKeywordsResponse(keywords=lines or [content[:200]])

@app.post("/generate/meta-description", response_model=MetaDescriptionResponse)
async def meta_description(request: Request, body: MetaDescriptionRequest):
    _rate_limited(request)
    user = f"Topic: {body.topic}. Keywords: {body.keywords or 'None'}. Generate a concise SEO meta description (120-160 chars)."
    content = await chat("You are an SEO copywriter. Output only the meta description text.", user)
    return MetaDescriptionResponse(meta_description=content[:200])

@app.post("/generate/product-description", response_model=ProductDescriptionResponse)
async def product_description(request: Request, body: ProductDescriptionRequest):
    _rate_limited(request)
    user = f"Product: {body.product_name}. Features: {body.features}. Tone: {body.tone}. Write a compelling description (200-400 words)."
    content = await chat("You are a product copywriter. Output only the description in Markdown.", user)
    return ProductDescriptionResponse(description=content)

@app.post("/generate/faq", response_model=FaqResponse)
async def faq_generator(request: Request, body: FaqRequest):
    _rate_limited(request)
    user = f"Topic: {body.topic}. Generate 5-8 FAQs, formatted as Q: question\nA: answer\n\n"
    content = await chat("You are a FAQ expert. Output in Q: / A: format, one per block.", user)
    faqs = []
    for block in content.split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 2 and lines[0].startswith("Q:") and lines[1].startswith("A:"):
            faqs.append(FaqItem(question=lines[0][2:].strip(), answer="\n".join(lines[1:])[2:].strip()))
    return FaqResponse(faqs=faqs[:8])

@app.post("/generate/social-post", response_model=SocialPostResponse)
async def social_post(request: Request, body: SocialPostRequest):
    _rate_limited(request)
    user = f"Topic: {body.post_topic}. Tone: {body.tone}. Platform: {body.platform}. Generate 3-5 posts, one per line."
    content = await chat("You are a social media expert. Output one post per line.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:5]
    return SocialPostResponse(posts=lines or [content[:500]])

@app.post("/generate/instagram-caption", response_model=InstagramCaptionResponse)
async def instagram_caption(request: Request, body: InstagramCaptionRequest):
    _rate_limited(request)
    user = f"Image topic: {body.image_topic}. Tone: {body.tone}. Emoji style: {body.emoji_style}. Generate one caption."
    content = await chat("You are an Instagram influencer. Output only the caption.", user)
    return InstagramCaptionResponse(caption=content[:300])

@app.post("/generate/youtube-script", response_model=YoutubeScriptResponse)
async def youtube_script(request: Request, body: YoutubeScriptRequest):
    _rate_limited(request)
    user = f"Video topic: {body.video_topic}. Audience: {body.audience or 'General'}. Tone: {body.tone}. Write a script (500-1000 words)."
    content = await chat("You are a YouTube scriptwriter. Output in sections like Intro, Body, Outro.", user)
    return YoutubeScriptResponse(script=content)

@app.post("/generate/youtube-thumbnail", response_model=YoutubeThumbnailResponse)
async def youtube_thumbnail(request: Request, body: YoutubeThumbnailRequest):
    _rate_limited(request)
    user = f"Video topic: {body.video_topic}. Tone: {body.tone}. Suggest 5 thumbnail titles, one per line."
    content = await chat("You are a YouTube expert. Output one title per line, no numbering.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:5]
    return YoutubeThumbnailResponse(titles=lines or [content[:200]])

@app.post("/generate/ai-image", response_model=AiImageResponse)
async def ai_image(request: Request, body: AiImageRequest):
    _rate_limited(request)
    style_hint = (body.style or "natural").strip().lower()
    full_prompt = f"{body.prompt.strip()}. Style: {style_hint}, high quality."
    try:
        response = await client.images.generate(
            model="dall-e-3",
            prompt=full_prompt[:4000],
            n=1,
            size="1024x1024",
            quality="standard",
            response_format="b64_json",
        )
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=500, detail="No image returned from API.")
        b64 = getattr(response.data[0], "b64_json", None)
        if not b64:
            raise HTTPException(status_code=500, detail="No image data in API response.")
        data_url = f"data:image/png;base64,{b64}"
        return AiImageResponse(image_url=data_url)
    except HTTPException:
        raise
    except Exception as e:
        msg = str(e).lower()
        if "content_policy" in msg or "safety" in msg or "inappropriate" in msg:
            raise HTTPException(status_code=400, detail="Prompt was rejected. Please try a different description.")
        if "billing" in msg or "insufficient" in msg or "quota" in msg:
            raise HTTPException(status_code=503, detail="Image generation is temporarily unavailable. Please try again later.")
        logger.exception("DALL-E image generation failed: %s", e)
        err_detail = str(e)
        if len(err_detail) > 200:
            err_detail = err_detail[:200] + "..."
        raise HTTPException(status_code=500, detail=f"Image generation failed: {err_detail}")


class DownloadImageBody(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048)


def _fetch_image_bytes(url: str) -> bytes:
    """Synchronous fetch for image URL (run in thread). Only allow OpenAI CDN."""
    if not url.startswith("https://"):
        raise ValueError("Invalid URL scheme")
    if "openai" not in url and "oaidalleapiprodscus" not in url:
        raise ValueError("URL must be from OpenAI image service")
    req = urllib.request.Request(url, headers={"User-Agent": "ArticleForge/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


@app.post("/api/download-image", include_in_schema=False)
async def download_image_proxy(body: DownloadImageBody):
    """Proxy image download so the browser gets a proper PNG file with correct filename."""
    try:
        data = await asyncio.to_thread(_fetch_image_bytes, body.url.strip())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.warning("Download proxy failed: %s", e)
        raise HTTPException(status_code=502, detail="Could not fetch image for download.")
    return Response(
        content=data,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="articleforge-image.png"'},
    )

@app.post("/generate/marketing-video", response_model=MarketingVideoResponse)
async def marketing_video(request: Request, body: MarketingVideoRequest):
    _rate_limited(request)
    user = f"Product: {body.product_name}. Features: {body.features}. Tone: {body.tone}. Write a 1-2 min video script."
    content = await chat("You are a video scriptwriter. Output in sections like Voiceover, Visuals.", user)
    return MarketingVideoResponse(script=content)

@app.post("/generate/cta", response_model=CtaResponse)
async def cta(request: Request, body: CtaRequest):
    _rate_limited(request)
    user = f"Action: {body.action}. Product/Service: {body.product_service}. Tone: {body.tone}. Generate 5 CTAs, one per line."
    content = await chat("You are a copywriter. Output one CTA per line, no numbering.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:5]
    return CtaResponse(ctas=lines or [content[:200]])

@app.post("/generate/email", response_model=EmailResponse)
async def email_copy(request: Request, body: EmailRequest):
    _rate_limited(request)
    user = f"Topic: {body.email_topic}. Audience: {body.audience or 'General'}. Tone: {body.tone}. Write a full email."
    content = await chat("You are an email marketer. Output in format: Subject: ...\nBody: ...", user)
    return EmailResponse(email=content)

@app.post("/generate/page-summary", response_model=PageSummaryResponse)
async def page_summary(request: Request, body: PageSummaryRequest):
    _rate_limited(request)
    content = await chat("Summarize the following in 3-5 sentences. Output only the summary.", body.content[:25000])
    return PageSummaryResponse(summary=content[:2000])

@app.post("/generate/competitor-analysis", response_model=CompetitorAnalysisResponse)
async def competitor_analysis(request: Request, body: CompetitorAnalysisRequest):
    _rate_limited(request)
    user = f"Competitor: {body.competitor_url}. Topic: {body.topic}. Suggest 8-15 content topics or keywords, one per line."
    content = await chat("You are an SEO strategist. Output one suggestion per line. No numbering.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:20]
    return CompetitorAnalysisResponse(suggestions=lines or [content[:200]])

@app.post("/generate/internal-links", response_model=InternalLinksResponse)
async def internal_links(request: Request, body: InternalLinksRequest):
    _rate_limited(request)
    user = f"Article excerpt:\n{body.article_content[:12000]}\n\nPages to link:\n{body.website_pages[:3000]}\nSuggest internal links (page - anchor), one per line."
    content = await chat("You are an SEO expert. Suggest internal links as 'page - anchor text', one per line.", user)
    lines = [l.strip() for l in content.splitlines() if l.strip()][:20]
    return InternalLinksResponse(links=lines or [content[:300]])

@app.post("/generate/sitemap", response_model=SitemapResponse)
async def generate_sitemap(request: Request, body: SitemapRequest):
    _rate_limited(request)
    url = body.website_url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="Website URL required.")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    user = f"Website: {url}. Output valid sitemap XML only: <?xml...><urlset> with <url><loc>...</loc><changefreq>...</changefreq><priority>...</priority></url>... </urlset>."
    content = await chat("You output only valid sitemap XML. No markdown or explanation.", user)
    if "<urlset" not in content:
        content = f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n  <url><loc>{url.rstrip("/")}/</loc><changefreq>weekly</changefreq><priority>1.0</priority></url>\n</urlset>'
    return SitemapResponse(sitemap=content[:50000])

@app.post("/generate/landing-page", response_model=LandingPageResponse)
async def landing_page(request: Request, body: LandingPageRequest):
    _rate_limited(request)
    if not body.product_name.strip() or not body.features.strip():
        raise HTTPException(status_code=422, detail="Product name and features required.")
    user = f"Product: {body.product_name}. Audience: {body.target_audience or 'General'}. Features: {body.features}. Tone: {body.tone}. Output: HEADLINE: ... SUBHEADLINE: ... FEATURES: (one per line) CTA: ..."
    content = await chat("Output landing page copy with labels HEADLINE:, SUBHEADLINE:, FEATURES:, CTA:. One item per line.", user)
    p = _parse_landing(content)
    return LandingPageResponse(headline=p["headline"], subheadline=p["subheadline"], features=p["features"], cta=p["cta"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
