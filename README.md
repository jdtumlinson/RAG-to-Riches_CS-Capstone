# RAG to Riches

## Introduction

- This project is a collaboration between Oregon State University's *"Senior Software Engineering Project"* class and SSOE Group, a globally recognized architecture and engineering firm. The project aims to leverage Retrieval Augmented Generation to enhance the submittal review process within the AEC industry. To learn more, please refer to our project proposal, *"proposal.pdf"*, located in the main repository.

## How to Run

### 1. Execute the Setup Script
To begin, ensure the `test.sh` script has execute permissions. If not, execute the following command to grant permissions:
   ```bash
   chmod +x test.sh
   ```

Next, execute the `test.sh` script to create and configure the virtual environment:
   ```bash
   bash test.sh
   ```

### 2. Activate the Virtual Environment
After executing the setup script, activate the virtual environment using the following command:
```bash
source test_env/bin/activate
```

### 3. Configure OpenAI API Key
Update `config.py` with your OpenAI API key to enable its use.

### 4. Launch the Application
With the virtual environment activated, launch the application by running:
```bash
python main.py
