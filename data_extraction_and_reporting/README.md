# Livesport Match Pipeline (Scrape → Structured Data → LLM Article)

This project is a small automation pipeline that:

1. Scrapes a Livesport match page (teams, score, basic match metadata).
2. Extracts the full event timeline (goals, cards, substitutions, VAR, etc.) by capturing XHR responses and parsing the Livesport/Flashscore `x/feed` format.
3. Saves the extracted data into structured outputs (`.json` + `.csv`).
4. Optionally calls an LLM to generate a **football journalist–style article** in the selected language.

## What gets generated

Everything is written to the `out/` directory:

- `match_data__<match-slug>.json`  
  Full structured match payload including extracted events.

- `match_events__<match-slug>.csv`  
  Events timeline in a tabular format (easy to analyze or import elsewhere).

- `report__<match-slug>__<lang>.md`  
  LLM-generated match article (or a local fallback report if you run with `--no-llm`).

- `out/xhr_dump/`  
  Debug artifacts: raw captured XHR/fetch responses used to extract timeline data.

## Requirements

- Python 3.10+ recommended
- Playwright with Chromium installed

## Installation

```bash
pip install playwright pandas python-dotenv requests beautifulsoup4 lxml
playwright install chromium
```

## Configuration

Create a `.env` file in the project root and add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

- `OPENAI_API_KEY` is required to generate the LLM article.
- If you only want scraping + JSON/CSV outputs, you can run with `--no-llm` and skip the key.

Do **not** commit `.env` to GitHub. Add it to `.gitignore`.

## Usage

### Scrape only (no LLM)

This mode is best for debugging extraction and verifying that the timeline is being parsed correctly.

```bash
python livesport_pipeline.py "MATCH_URL" --no-llm
```

Run with a visible browser window (useful if the page is dynamic and you want to see what loads):

```bash
python livesport_pipeline.py "MATCH_URL" --no-llm --headed
```

### Generate the LLM article

English:

```bash
python livesport_pipeline.py "MATCH_URL" --lang en
```

Czech:

```bash
python livesport_pipeline.py "MATCH_URL" --lang cs
```

Example:

```bash
python livesport_pipeline.py "https://www.livesport.cz/zapas/fotbal/sparta-praha-6qA358jH/zbrojovka-brno-4d5TT6i5/?mid=StyIyb9D#/prehled-zapasu/prehled-zapasu" --lang en
```

## CLI options

- `--no-llm`  
  Skip the LLM call. Still generates JSON/CSV and a simple local Markdown report.

- `--headed`  
  Runs Playwright in headed mode (browser UI visible).

- `--lang <code>`  
  Output language for the generated article. Examples: `en`, `cs`, `de`, `es`, `fr`.  
  Default: `en`.

## How it works

- Playwright loads the match page and navigates to the match summary route.
- The script listens to XHR/fetch responses and stores them in `out/xhr_dump/`.
- The event timeline is typically delivered via a Livesport/Flashscore feed format (`x/feed/df_sui_1_<matchId>`).  
  The pipeline parses this feed into a structured event list.
- Events are normalized into a consistent schema:
  - `minute`
  - `side` (home/away)
  - `event_type` (goal, yellow_card, substitution, etc.)
  - `player`
  - `detail` (assist/sub info when available)
  - `raw_text` (robust fallback text)
- If LLM mode is enabled, the script sends the structured JSON to an LLM and requests a ~300 word football journalist–style article plus a full timeline section.

## Troubleshooting

### No events extracted

1. Check `out/xhr_dump/` and search inside dumped files for feed markers like:
   - `~III`
   - `IK÷`
   - `IB÷`
2. If the timeline is present in the dump but not parsed, the feed format may have changed.  
   The dumps are intentionally saved so you can adjust parsing without guessing.

### OpenAI errors (quota / billing)

If you see errors such as `insufficient_quota`, your OpenAI account/project has no usable quota.  
Add billing or increase limits, then re-run.

## API key safety

- Keep `OPENAI_API_KEY` in `.env`
- Never hardcode it in the script
- Never commit `.env` into version control
- Sharing the code is safe as long as you do not share the key
