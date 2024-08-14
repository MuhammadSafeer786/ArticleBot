# Article QA with LLMs and Streamlit

This project leverages Large Language Models (LLMs) to create a practical tool that allows users to input the URL of any online article and ask questions related to it. This is particularly useful for news researchers working with financial articles, global news, or any other specialized content.

## Features

- **Data Loading**: Uses LangChain's `unstructured_data_loader` to fetch and process the article's content.
- **Data Splitting**: Optimizes the usage of OpenAI API tokens by splitting the article into manageable chunks.
- **Vector Database**: Implements FAISS to create a vector database for faster and more efficient searches.
- **Streamlit Integration**: The entire application is built with Streamlit, providing a user-friendly interface.

## Setup
### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/article-qa-llm.git
    cd article-qa-llm
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the root directory and add your OpenAI API key:
    ```bash
    OPENAI_API_KEY=your_openai_api_key_here
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Usage

1. Input the URL of an online article.
2. The tool will load and process the article using the LangChain `unstructured_data_loader`.
3. The article is split into chunks to save OpenAI API tokens.
4. A vector database is created using FAISS for faster search and retrieval.
5. You can now ask questions related to the article, and the model will provide answers based on the content.

## Acknowledgments

- Huge thanks to the [codebasics YouTube channel](https://www.youtube.com/c/codebasics) for their amazing tutorials that helped me build this project.
