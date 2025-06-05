# Flow Design Document

## Overview
This document outlines the design patterns and specifications for PocketFlow flows in the claude-pocketflow template.

## Flow Architecture

### Core Concepts
1. **Flows** are directed graphs of nodes
2. **Nodes** are atomic units of work
3. **Transitions** are action-based connections between nodes
4. **Store** is the shared state passed between nodes

### Flow Definition Structure
```python
flow_definition = {
    "node_id": {
        "node": "NodeClassName",
        "transitions": {
            "action_name": "next_node_id",
            "error": "error_handler_id"
        }
    }
}
```

## Standard Flow Patterns

### 1. Linear Flow
Simple sequential execution.
```
Start → Process → Validate → Complete
```

### 2. Conditional Flow
Branching based on conditions.
```
Start → Check → [Branch A | Branch B] → Merge → Complete
```

### 3. Error Handling Flow
Robust error recovery.
```
Start → Process → [Success → Complete | Error → Retry → Process]
```

### 4. Parallel Flow
Concurrent execution paths.
```
Start → Split → [Path A & Path B] → Join → Complete
```

## Example Flows

### Basic Data Processing Flow
```python
data_processing_flow = {
    "start": {
        "node": "DataLoaderNode",
        "transitions": {
            "loaded": "validate",
            "error": "handle_load_error"
        }
    },
    "validate": {
        "node": "DataValidatorNode",
        "transitions": {
            "valid": "process",
            "invalid": "handle_validation_error",
            "error": "handle_error"
        }
    },
    "process": {
        "node": "DataProcessorNode",
        "transitions": {
            "complete": "save",
            "error": "handle_process_error"
        }
    },
    "save": {
        "node": "DataSaverNode",
        "transitions": {
            "saved": "end",
            "error": "handle_save_error"
        }
    }
}
```

### API Integration Flow
```python
api_flow = {
    "start": {
        "node": "AuthenticationNode",
        "transitions": {
            "authenticated": "fetch_data",
            "failed": "handle_auth_error"
        }
    },
    "fetch_data": {
        "node": "APIFetchNode",
        "transitions": {
            "success": "transform",
            "rate_limited": "wait_and_retry",
            "error": "handle_api_error"
        }
    },
    "transform": {
        "node": "DataTransformNode",
        "transitions": {
            "complete": "store",
            "error": "handle_transform_error"
        }
    }
}
```

## Node Design Guidelines

### 1. Single Responsibility
Each node should do one thing well.

### 2. Idempotency
Nodes should produce the same result when run multiple times with the same input.

### 3. Error Handling
Every node should handle its specific errors and set appropriate actions.

### 4. State Management
- Input validation in prep()
- Core logic in exec()
- Cleanup in post()

## Store Design Patterns

### Namespacing
Use prefixes to avoid key collisions:
```python
store = {
    "user_input": {...},
    "api_response": {...},
    "transform_output": {...}
}
```

### Status Tracking
Maintain flow state:
```python
store = {
    "flow_status": "processing",
    "current_step": "validate",
    "error_count": 0,
    "start_time": "2024-01-01T00:00:00Z"
}
```

### Error Context
Preserve error information:
```python
store = {
    "last_error": {
        "node": "APIFetchNode",
        "message": "Connection timeout",
        "timestamp": "2024-01-01T00:00:00Z",
        "retry_count": 2
    }
}
```

## Testing Flows

### Unit Testing Nodes
Test each node in isolation:
```python
def test_node_happy_path():
    node = MyNode()
    store = {"input": "test"}
    result = node.exec(store)
    assert result["status"] == "success"
```

### Integration Testing Flows
Test complete flow execution:
```python
def test_flow_end_to_end():
    flow = MyFlow()
    result = flow.run({"user_id": "123"})
    assert "output" in result
    assert result["flow_status"] == "complete"
```

## Performance Considerations

### 1. Async Operations
Use async nodes for I/O operations:
```python
class AsyncAPINode:
    async def exec(self, store):
        response = await fetch_data()
        store["data"] = response
        return store
```

### 2. Resource Management
Clean up resources in post():
```python
def post(self, store):
    if "temp_file" in store:
        os.remove(store["temp_file"])
    return store
```

### 3. Caching
Implement caching for expensive operations:
```python
def exec(self, store):
    cache_key = f"data_{store['user_id']}"
    if cached := cache.get(cache_key):
        store["data"] = cached
    else:
        data = expensive_operation()
        cache.set(cache_key, data)
        store["data"] = data
    return store
```

## Monitoring and Debugging

### Logging
Each node should log its execution:
```python
import logging

logger = logging.getLogger(__name__)

def exec(self, store):
    logger.info(f"Processing user {store.get('user_id')}")
    # ... processing ...
    logger.info(f"Completed with status {store.get('status')}")
    return store
```

### Metrics
Track key performance indicators:
- Node execution time
- Flow completion rate
- Error frequency
- Retry counts

### Debugging
Use store snapshots:
```python
def debug_flow(flow, initial_store):
    snapshots = []
    # Hook into flow execution to capture snapshots
    result = flow.run(initial_store, snapshot_callback=snapshots.append)
    return result, snapshots
```