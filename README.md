

### Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Your Free API Key

1. Visit [openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up (completely free)
3. Create a new API key
4. Copy your key (starts with `sk-or-v1-...`)

### 3. Set Your API Key

**Option A: Environment Variable (Recommended)**
```bash
# Windows
set OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Mac/Linux
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

**Option B: Edit the Code**
Open `pdf_chat_app.py` and replace line 19:
```python
OPENROUTER_API_KEY = "XXXXX_API_KEY_XXXXX"  # Replace with your actual key
```

### 4. Run the App

```bash
python pdf_chat_app.py
```

### 5. Open Your Browser

Navigate to: **http://localhost:5000**

## How to Use

### Upload Documents
1. **Drag and drop** PDFs into the upload zone, OR
2. **Click** the upload zone to browse files
3. Supported formats: PDF, TXT, CSV, JSON, MD
4. Multiple files can be uploaded at once

### Ask Questions
1. Type your question in the input box
2. Press **Enter** to send (or click the send button)
3. Use **Shift+Enter** for new lines
4. Wait for the AI to analyze and respond

### Quick Examples
Click any suggestion button in the sidebar:
- Summarize the document
- Extract key dates
- Find all names
- Identify main topics
- Get numbers & stats

### Switch Models
- Use the dropdown in the top-right to change models
- Try different models for different results
- The app automatically saves your preference

## Example Queries

### Document Analysis
```
"Summarize the main points of this PDF"
"What are the key takeaways?"
"Give me a 3-sentence overview"
```

### Information Extraction
```
"List all dates mentioned in chronological order"
"Extract all email addresses and phone numbers"
"Find all numerical data and statistics"
"Who are the main people mentioned?"
```

### Search & Find
```
"Does this document mention climate change?"
"Find all references to 'budget' or 'cost'"
"What does it say about renewable energy?"
```

### Comparison
```
"Compare the budgets between 2023 and 2024"
"What's the difference between section 2 and section 3?"
"Which recommendations appear in both documents?"
```

## Configuration

### Available Free Models

The app automatically fetches the latest free models from OpenRouter. Common ones include:

- **Llama 3.2 3B** - Fast, good quality
- **Llama 3.2 1B** - Fastest, lighter
- **Gemma 2 9B** - Higher quality
- **Qwen 2 7B** - Good alternative

### File Limits

```python
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB per file
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB total
```

### Supported File Types

```python
ALLOWED_EXT = {".pdf", ".txt", ".csv", ".json", ".md"}
```

