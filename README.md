# 🔍 WhatsApp Analyzer

Advanced pattern detection system for WhatsApp conversations. Analyzes exported chats to identify suspicious behaviors through risk-classified term dictionaries.

## 📋 Supported Detection Categories

- **Sexual Harassment**
- **Cyberbullying**
- **Threats and Violence**
- **Drugs**
- **Infidelity**
- **Profanity (Argentina)**
- **Theft and Fraud**
- **Suicide and Self-Harm**
- **Complete** (all categories combined)

Categories can be combined and extended with custom terms.

## 🚀 Installation

### Requirements

- Python 3.9+

### Setup

```bash
git clone <repository-url>
cd whatsapp-scanner
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run whatsapp_analyzer_streamlit.py
```

App opens automatically at `http://localhost:8501`.

## 📱 Export WhatsApp Chats

### Android

1. Open WhatsApp
2. Go to the chat you want to analyze
3. Tap the three dots (⋮) → **More** → **Export chat**
4. Select **"Without media"** ⚠️ **Important**
5. Save the `.txt` file

### iPhone

1. Open WhatsApp
2. Go to the chat you want to analyze
3. Tap the contact/group name
4. Scroll down → **Export chat**
5. Select **"Without media"** ⚠️ **Important**
6. Save the `.txt` file

## 🎯 Using the Application

### 1. Select Detection Categories

In the sidebar, **"🎯 Detection Type(s)"** is a multi-select: you can choose one or multiple predefined categories at once (e.g., "Drugs" + "Threats and Violence"). Their terms are combined without duplication.

### 2. (Optional) Add Custom Terms

In **"📁 Add Custom Terms"** you can:
- **Type terms:** Enter names or words separated by commas or line breaks
- **Upload file:** Upload a `.csv` or `.txt` with format `term,category`

Custom terms are added to selected categories (don't replace them). If no categories are selected, the custom file works as the complete dictionary.

### 3. Configure Sensitivity

- **Low** (threshold 0.75): fewer false positives, detects only clear cases
- **Medium** (threshold 0.60): balanced (recommended)
- **High** (threshold 0.45): more sensitive, may have more false positives

You can also set a **custom threshold** (0.0 to 1.0) that overrides the sensitivity level.

### 4. Upload Chat File

Upload the `.txt` file you exported. The app automatically extracts date, sender, and message from each line.

### 5. Review Results

- **Statistics:** total messages, detections, percentage, average risk
- **Visualizations:** risk distribution and detections by sender
- **Evidence:** filterable list (by sender and minimum risk) of detected messages with matched terms

### 6. Download Reports

- **Complete CSV:** all messages with their scores
- **Detections CSV:** only messages marked as `DETECTED`
- **Executive Report:** summary and disclaimer

## 📖 Custom Dictionary Format

`.csv` or `.txt` file, one term per line:

```
term,category
```

### Recognized Categories

| File Category | Classification |
|---|---|
| `palabras_alta` | High risk |
| `palabras_media` | Medium risk |
| `frases_contexto` | Context phrase |
| `contexto_laboral`, `contexto_relacion`, `contexto_financiero`, `contexto_agresion`, `contexto_emocional`, `contexto_digital`, `contexto_sustancias` | Context (any of these) |

Empty lines and lines starting with `#` are ignored (useful for comments).

### Example

```csv
# School bullying dictionary
stupid,palabras_alta
idiot,palabras_alta
nobody likes you,frases_contexto
school,contexto_digital
```

## ⚙️ Risk Calculation

For each message, the app counts matches in each category (high risk, medium risk, context phrases, context). That count is converted to a ratio that **saturates at 3 matches** (independent of dictionary size), multiplied by the category weight according to sensitivity, and summed with bonuses when high-risk + context or high-risk + work context combine. The final result is between 0.0 and 1.0.

A message is marked `DETECTED` when its score exceeds the configured threshold.

## 🛡️ Legal and Ethical Considerations

- Obtain consent before analyzing private conversations of others
- Respect local privacy laws
- Use only for legitimate purposes (research, safety, protection)
- All files are processed locally in memory — the app doesn't persist data on disk or send it to external servers
- Results are indicative, not definitive proof: false positives and false negatives can occur. Always verify evidence manually

## 🔧 Troubleshooting

**File not loading / messages not extracting:**
- Verify it's a `.txt` file exported from WhatsApp with "Without media" option
- Expected format per line: similar to `12/06/24, 10:30 a. m. - Name: message` (or variants with brackets/AM-PM)
- By design, the app requires 5+ messages to recognize the export pattern

**Many false positives:**
- Lower sensitivity to "Low" or increase custom threshold

**Few results:**
- Raise sensitivity to "High" or lower custom threshold
- Check that selected categories (or custom dictionary) include relevant terms

## 🆘 Crisis Resources

- 🚨 **Argentina**: 135 (Centro de Asistencia al Suicida)
- 🚨 **México**: 800-290-0024 (SAPTEL)
- 🚨 **España**: 717-003-717 (Teléfono de la Esperanza)
- 🚨 **Chile**: 4141 (Salud Responde)

## 📁 Project Structure

```
whatsapp-scanner/
├── src/
│   ├── __init__.py
│   ├── parser.py          # Message extraction and parsing
│   ├── dictionary.py      # Dictionary loading and merging
│   ├── analyzer.py        # Risk scoring
│   └── ui.py              # UI helpers
├── data/                  # Predefined dictionaries
├── scripts/
│   └── convert_dictionaries.py
├── tests/                 # Unit tests
├── whatsapp_analyzer_streamlit.py  # Main application
├── requirements.txt
└── README.md
```

## 🧪 Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

## 🔒 Privacy

All files are processed locally in memory. No data is stored or transmitted to external servers. Your results are yours alone.

---

**This software is provided for educational and research purposes. The user is responsible for ethical and legal use of the tool.**
