# API Reference

## Node Base Class

### AbstractNode
Base interface that all nodes should implement.

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class AbstractNode(ABC):
    @abstractmethod
    def prep(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and validate inputs."""
        pass
    
    @abstractmethod
    def exec(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Execute main logic."""
        pass
    
    @abstractmethod
    def post(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process and cleanup."""
        pass
```

## Built-in Nodes

### InitNode
Initializes the flow with default values.

**Store Keys:**
- Input: None required
- Output: 
  - `initialized`: bool
  - `timestamp`: str
  - `flow_id`: str

**Example:**
```python
node = InitNode()
store = {}
store = node.prep(store)
store = node.exec(store)
store = node.post(store)
# store now contains initialization data
```

### ValidationNode
Validates store data against schemas.

**Store Keys:**
- Input:
  - `data`: Dict[str, Any] - Data to validate
  - `schema`: Dict[str, Any] - JSON schema
- Output:
  - `validation_result`: bool
  - `validation_errors`: List[str]

### ErrorHandlerNode
Handles errors and determines retry strategy.

**Store Keys:**
- Input:
  - `error`: str - Error message
  - `retry_count`: int - Current retry attempt
- Output:
  - `action`: str - "retry", "fail", or "continue"
  - `retry_delay`: int - Seconds to wait before retry

## Flow Classes

### BaseFlow
Base class for all flows.

```python
class BaseFlow:
    def __init__(self):
        self.flow_definition = {}
        self.nodes = {}
    
    def register_node(self, name: str, node: AbstractNode) -> None:
        """Register a node instance."""
        pass
    
    def run(self, initial_store: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the flow."""
        pass
```

### FlowBuilder
Utility for building flows programmatically.

```python
builder = FlowBuilder()
builder.add_node("start", InitNode())
builder.add_node("validate", ValidationNode())
builder.add_transition("start", "success", "validate")
builder.add_transition("validate", "error", "error_handler")
flow = builder.build()
```

## Utility Classes

### APIClient
Base class for API integrations.

```python
class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
    
    async def request(
        self, 
        method: str, 
        endpoint: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Make an API request."""
        pass
```

### RetryableClient
Wrapper that adds retry logic to any client.

```python
client = APIClient(base_url, api_key)
retryable = RetryableClient(
    client,
    max_retries=3,
    backoff_factor=2
)
```

## Store Utilities

### StoreValidator
Validates store contents.

```python
validator = StoreValidator()
validator.require_keys(store, ["user_id", "action"])
validator.validate_types(store, {"user_id": str, "action": str})
```

### StoreSnapshot
Creates and restores store snapshots.

```python
snapshot = StoreSnapshot()
backup = snapshot.create(store)
# ... modifications to store ...
restored = snapshot.restore(backup)
```

## Testing Utilities

### NodeTestCase
Base class for node tests.

```python
class TestMyNode(NodeTestCase):
    def test_happy_path(self):
        node = MyNode()
        store = {"input": "value"}
        result = self.run_node(node, store)
        self.assert_store_contains(result, "output")
```

### FlowTestCase
Base class for flow tests.

```python
class TestMyFlow(FlowTestCase):
    def test_complete_flow(self):
        flow = MyFlow()
        result = self.run_flow(flow, {"user": "test"})
        self.assert_flow_success(result)
```

## Common Store Keys

### Standard Keys
- `status`: Current status of the operation
- `error`: Error message if any
- `timestamp`: When the operation occurred
- `user_id`: User identifier
- `request_id`: Unique request identifier

### Action Keys
- `action`: Next action to take
- `next_node`: Explicit next node (overrides flow)
- `retry_count`: Number of retries attempted
- `retry_delay`: Delay before next retry

### Data Keys
- `input_data`: Original input
- `processed_data`: Transformed data
- `output_data`: Final output
- `metadata`: Additional information

## Error Codes

### Node Errors
- `NODE_VALIDATION_ERROR`: Input validation failed
- `NODE_EXECUTION_ERROR`: Error during exec()
- `NODE_TIMEOUT`: Node execution timeout

### Flow Errors
- `FLOW_INVALID_TRANSITION`: No valid transition
- `FLOW_MAX_RETRIES`: Exceeded retry limit
- `FLOW_CIRCULAR_REFERENCE`: Circular flow detected

### Utility Errors
- `API_CONNECTION_ERROR`: Cannot reach service
- `API_AUTHENTICATION_ERROR`: Invalid credentials
- `API_RATE_LIMIT`: Rate limit exceeded

## Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=sk-...
WEATHER_API_KEY=...

# Flow Configuration
MAX_RETRIES=3
RETRY_BACKOFF=2
NODE_TIMEOUT=30

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Flow Configuration
```python
flow_config = {
    "max_retries": 3,
    "timeout": 300,
    "parallel_execution": True,
    "store_backend": "memory"  # or "redis"
}
```