# PocketFlow-Specific Patterns and Rules

## Core PocketFlow Concepts

### 1. Node Architecture
Nodes are the fundamental building blocks with three lifecycle methods:

```python
class NodeTemplate:
    def prep(self, store: dict) -> dict:
        """
        Preparation phase:
        - Validate inputs from store
        - Set up required resources
        - Check preconditions
        Returns: Updated store with prep results
        """
        pass
    
    def exec(self, store: dict) -> dict:
        """
        Execution phase:
        - Perform main logic
        - Call utilities as needed
        - Generate results
        Returns: Store with execution results
        """
        pass
    
    def post(self, store: dict) -> dict:
        """
        Post-processing phase:
        - Clean up resources
        - Format outputs
        - Update store for next nodes
        Returns: Final store state
        """
        pass
```

### 2. Flow Patterns

#### Action-Based Transitions
```python
flow_definition = {
    "start": {
        "node": "InitNode",
        "transitions": {
            "success": "process",
            "error": "handle_error"
        }
    },
    "process": {
        "node": "ProcessNode",
        "transitions": {
            "complete": "finalize",
            "retry": "process",
            "fail": "handle_error"
        }
    }
}
```

#### Shared Store Pattern
- All nodes share a common store dictionary
- Store maintains state throughout flow execution
- Keys should be namespaced to avoid conflicts
- Store includes both inputs and outputs

### 3. Utility Organization

Each utility file in `utils/` should:
- Handle one specific external service or API
- Include error handling and retries
- Provide a clean interface for nodes
- Include comprehensive logging

Example structure:
```python
# utils/openai_client.py
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat_completion(self, messages: list) -> Dict[str, Any]:
        """Make a chat completion request"""
        try:
            # Implementation
            pass
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

# Testing function (replaced by pytest in our setup)
def main():
    """Test the OpenAI client"""
    client = OpenAIClient("test-key")
    # Test implementation
```

### 4. Self-Evaluation Nodes

For uncertain outputs, implement self-evaluation:
```python
class SelfEvaluatingNode:
    def exec(self, store: dict) -> dict:
        result = self._generate_output(store)
        evaluation = self._evaluate_output(result)
        
        if evaluation["confidence"] < 0.8:
            store["needs_review"] = True
            store["evaluation"] = evaluation
        
        store["result"] = result
        return store
```

### 5. Context Management

- Keep context minimal and relevant
- Use RAG for retrieving historical context
- Implement context windows for long conversations
- Clear context between unrelated flows

### 6. Flow Design Principles

1. **Start Simple**: Begin with minimal viable flow
2. **Iterative Enhancement**: Add complexity gradually
3. **Clear Actions**: Each node should have clear success/failure paths
4. **Modular Design**: Nodes should be reusable across flows
5. **State Management**: Use store effectively for passing data

### 7. Error Handling Strategy

```python
class RobustNode:
    def exec(self, store: dict) -> dict:
        try:
            result = self._process(store)
            store["status"] = "success"
            store["result"] = result
        except ValidationError as e:
            store["status"] = "validation_error"
            store["error"] = str(e)
        except ExternalAPIError as e:
            store["status"] = "retry"
            store["error"] = str(e)
            store["retry_count"] = store.get("retry_count", 0) + 1
        except Exception as e:
            store["status"] = "fatal_error"
            store["error"] = str(e)
        
        return store
```

### 8. Agent Response Format

When implementing agentic patterns, use structured responses:
```yaml
thinking: |
  Analyzing the user request...
  Need to check prerequisites first...
action: process_request
parameters:
  input: user_data
  mode: strict
  validate: true
```

### 9. Flow Optimization Tips

- Profile node execution times
- Implement caching where appropriate
- Use async operations for I/O-bound tasks
- Monitor memory usage in shared store
- Add instrumentation for debugging

### 10. Reliability Patterns

- Implement retry logic with exponential backoff
- Add circuit breakers for external services
- Use timeouts for all external calls
- Maintain audit logs for flow execution
- Implement graceful degradation