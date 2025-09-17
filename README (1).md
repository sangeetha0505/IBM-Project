# Citizen AI – Intelligent Citizen Engagement Platform  

## 📌 Introduction  
Citizen AI is a platform built using **IBM Granite models** to provide quick and helpful answers about government services, civic issues, and city safety analysis. It also includes features for **citizen interaction, public sentiment tracking, and dashboard visualization** to support government transparency and engagement.  

The system runs efficiently on **Google Colab with GPU support**, making it lightweight, cost-effective, and easy to deploy.  

---

## 🚀 Features  
- **Conversational Assistance** – Provides natural language responses to citizen queries.  
- **City Analysis** – Generates reports on crime index, accidents, and overall safety of cities.  
- **Sentiment Tracking** – Analyzes citizen feedback for government officials.  
- **Dashboard Visualization** – Displays insights in a simple and transparent way.  
- **Lightweight Deployment** – Runs seamlessly in Google Colab.  

---

## 🏗️ Architecture  
- **Frontend**: [Gradio](https://gradio.app/) – Interactive UI for city analysis and citizen queries.  
- **Backend**: [IBM Granite Models via Hugging Face](https://huggingface.co/ibm-granite) – Provides natural language understanding and responses.  
- **Deployment**: Google Colab (with GPU) – Low-cost and reliable hosting.  
- **Version Control**: GitHub – For project collaboration and updates.  

---

## 👨‍💻 Team Members  
- **Sangeetha B**  
- **Yuva Lakshmi N**  
- **Rexlin Mary W**  
- **Vaishnavi N**  

---

## ⚙️ Setup Instructions  

### Prerequisites  
- Python 3.9 or later  
- Gradio Framework  
- Hugging Face access to IBM Granite Models  
- Git installed  
- Google Colab account with T4 GPU  

### Installation  
1. Open **Google Colab** and create a new notebook.  
2. Change runtime → **GPU (T4)**.  
3. Install dependencies:  
   ```bash
   !pip install transformers torch gradio
   ```  
4. Copy the contents of `citizen_ai.py` into the notebook.  
5. Run the notebook cells sequentially.  
6. Access the **Gradio app link** generated.  
7. Push project files to GitHub for version control.  

---

## 📂 Folder Structure  
```
app/            # Gradio application scripts
notebooks/      # Google Colab notebooks
models/         # Hugging Face Granite model references
.github/        # GitHub configuration files
citizen_ai.py   # Main application file
requirements.txt# Dependencies list
```

---

## ▶️ Running the Application  
1. Open Google Colab notebook.  
2. Install required dependencies.  
3. Run all code cells.  
4. Launch Gradio app → get public link.  
5. Interact with **City Analysis** or **Citizen Services** tabs.  
6. Officials can monitor dashboards and citizen feedback.  

---

## 📜 Code (citizen_ai.py)  

```python
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

def city_analysis(city_name):
    prompt = f"Provide a detailed analysis of {city_name} including:\n1. Crime Index and safety statistics\n2. Accident rates and traffic safety information\n3. Overall safety assessment\n\nCity: {city_name}\nAnalysis:"
    return generate_response(prompt, max_length=1000)

def citizen_interaction(query):
    prompt = f"As a government assistant, provide accurate and helpful information about the following citizen query related to public services, government policies, or civic issues:\n\nQuery: {query}\nResponse:"
    return generate_response(prompt, max_length=1000)

# Create Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# City Analysis & Citizen Services AI")

    with gr.Tabs():
        with gr.TabItem("City Analysis"):
            with gr.Row():
                with gr.Column():
                    city_input = gr.Textbox(
                        label="Enter City Name",
                        placeholder="e.g., New York, London, Mumbai...",
                        lines=1
                    )
                    analyze_btn = gr.Button("Analyze City")

                with gr.Column():
                    city_output = gr.Textbox(label="City Analysis (Crime Index & Accidents)", lines=15)

            analyze_btn.click(city_analysis, inputs=city_input, outputs=city_output)

        with gr.TabItem("Citizen Services"):
            with gr.Row():
                with gr.Column():
                    citizen_query = gr.Textbox(
                        label="Your Query",
                        placeholder="Ask about public services, government policies, civic issues...",
                        lines=4
                    )
                    query_btn = gr.Button("Get Information")

                with gr.Column():
                    citizen_output = gr.Textbox(label="Government Response", lines=15)

            query_btn.click(citizen_interaction, inputs=citizen_query, outputs=citizen_output)

app.launch(share=True)
```

---

## 🔐 Authentication (Future Work)  
- API Keys  
- OAuth2 with IBM Cloud credentials  
- Role-based access for citizens and officials  

---

## 🧪 Testing  
- **Unit Testing** – Check code cells in Colab.  
- **Manual Testing** – Validate chatbot responses and outputs.  
- **Deployment Testing** – Run app on Colab with GPU.  

---

## ⚠️ Known Issues  
- Limited to Colab session runtime.  
- Requires stable internet connection.  
- Dependent on Hugging Face model availability.  

---

## 🔮 Future Enhancements  
- Dedicated cloud deployment (IBM Cloud, AWS, Azure).  
- Multi-language citizen support.  
- Integration with real government data sources.  
- Advanced dashboards with analytics.  
