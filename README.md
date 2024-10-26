# Prerequisites
Ensure the following are installed and configured before proceeding:

- PostgreSQL: You can install PostgreSQL locally or run it using Docker.
- Ollama: For users working with a local model, download it with:
```bash
ollama pull vaibhav-s-dabhade/phi-3.5-mini-instruct
```
- Database Migration: Run the migration script on PostgreSQL
- Hugging Face API Key: Required only if using a Hugging Face model.

# Setup Instructions
After cloning the repository, follow these steps:

- Create a folder for saving LLM plot outputs (plots)
```bash
mkdir plots
```

- Create a Virtual Environment:

```bash
python3 -m venv env
```
- Activate the Virtual Environment and install the dependencies:
```bash
source env/bin/activate      # On macOS/Linux
env\Scripts\activate         # On Windows
pip install -r requirements.txt
```
- Configure Environment Variables: Create a .env file in the root directory with the following variables. Fill in with your actual credentials where necessary:

```plaintext
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=localhost
HF_SECRET=your_huggingface_api_key  # Required if using Hugging Face
```

- Run the Application: Start the app with Streamlit:
```bash
streamlit run db_chatbot.py
```

# Additional Notes
- If using another LLM inference client, such as llama-cpp, update the code in expiriment.ipynb accordingly.
- The main db_chatbot.py file is currently configured to work with Hugging Face. Modify the `get_llm_output` function if working with another LLM inference client.
- For testing, expiriment.ipynb includes a small dummy dataframe in the Random Data Trail section, which can used if any issues are encountred with database connectivity
- Any plots generated by the LLM will be saved to the plots folder
