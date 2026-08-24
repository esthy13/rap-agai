## Exercise 2: Memory

Check the other README.md's for more information about the exercise structure. For exercise 2, issues are under the Milestone 2.

You can again use the model via the API. For this task, I provide an embedding model. Checkout the `embedding_example.py`.

```python
import os

from dotenv import load_dotenv
from agentscope.embedding import OpenAITextEmbedding
from agentic_ai_exercise import ENV_PATH, QWEN3_06B_Embed, QWEN3_VL_4B_Instruct

# Load API key and base URL from .env
load_dotenv(ENV_PATH)

api_key  = os.environ["LLM_API_KEY"]
api_base = os.environ["LLM_BASE_URL"]

# --- Embedding model (standalone, used outside the agent) ---
embedding_model = OpenAITextEmbedding(
    model_name=QWEN3_06B_Embed,
    api_key=api_key,
    base_url=api_base,           # note: base_url, not client_args
    dimensions=None,
)
```
If you use another class for calling this API. Make sure that the argument `"encoding_format": "float"` is set. If you want to set the dimensions, you will get an error. If your approach requries a different embedding size, you need to create a workaround.


