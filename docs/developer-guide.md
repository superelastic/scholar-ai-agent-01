# Developer Guide

This guide explains how to build agentic applications using the claude-pocketflow template.

## Overview

claude-pocketflow combines PocketFlow's node/flow architecture with professional Python development practices. It's designed for AI-assisted development using Cursor, Claude-Code, and Task-Master.

## Core Concepts

### 1. Nodes
Nodes are the fundamental building blocks of your application. Each node follows a three-phase lifecycle:

- **prep()**: Validate inputs and prepare resources
- **exec()**: Execute the main logic
- **post()**: Clean up and format outputs

### 2. Flows
Flows orchestrate nodes using action-based transitions. They define:
- The sequence of node execution
- Transition conditions between nodes
- Error handling paths

### 3. Shared Store
All nodes share a common store (dictionary) that maintains state throughout flow execution.

## Creating Your First Node

### Basic Node Structure
```python
# nodes.py
from typing import Dict, Any

class GreetingNode:
    """A simple node that generates a greeting."""
    
    def prep(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that we have a name in the store."""
        if "name" not in store:
            store["error"] = "Missing required field: name"
            store["status"] = "error"
        else:
            store["status"] = "ready"
        return store
    
    def exec(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the greeting."""
        if store.get("status") == "ready":
            name = store["name"]
            store["greeting"] = f"Hello, {name}!"
            store["status"] = "success"
        return store
    
    def post(self, store: Dict[str, Any]) -> Dict[str, Any]:
        """Clean up temporary data."""
        # Remove any temporary keys if needed
        store.pop("status", None)
        return store
```

### Testing Your Node
```python
# tests/test_nodes.py
import pytest
from nodes import GreetingNode

def test_greeting_node_success():
    node = GreetingNode()
    store = {"name": "Alice"}
    
    store = node.prep(store)
    assert store["status"] == "ready"
    
    store = node.exec(store)
    assert store["greeting"] == "Hello, Alice!"
    
    store = node.post(store)
    assert "status" not in store

def test_greeting_node_missing_name():
    node = GreetingNode()
    store = {}
    
    store = node.prep(store)
    assert store["status"] == "error"
    assert "Missing required field" in store["error"]
```

## Creating Flows

### Basic Flow Definition
```python
# flow.py
from typing import Dict, Any

class GreetingFlow:
    def __init__(self):
        self.flow_definition = {
            "start": {
                "node": "GreetingNode",
                "transitions": {
                    "success": "end",
                    "error": "handle_error"
                }
            },
            "handle_error": {
                "node": "ErrorHandlerNode",
                "transitions": {
                    "resolved": "start",
                    "failed": "end"
                }
            }
        }
    
    def run(self, initial_store: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the flow with the given initial store."""
        # Flow execution logic here
        pass
```

## Working with Utilities

### Creating a Utility
Each external service should have its own utility file:

```python
# utils/weather_api.py
import os
from typing import Dict, Any
import httpx

class WeatherAPI:
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.base_url = "https://api.weather.com/v1"
    
    async def get_weather(self, city: str) -> Dict[str, Any]:
        """Fetch weather data for a city."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/weather",
                params={"city": city, "key": self.api_key}
            )
            response.raise_for_status()
            return response.json()
```

### Using Utilities in Nodes
```python
class WeatherNode:
    def __init__(self):
        self.weather_api = WeatherAPI()
    
    async def exec(self, store: Dict[str, Any]) -> Dict[str, Any]:
        city = store.get("city")
        try:
            weather_data = await self.weather_api.get_weather(city)
            store["weather"] = weather_data
            store["status"] = "success"
        except Exception as e:
            store["error"] = str(e)
            store["status"] = "error"
        return store
```

## Best Practices

### 1. State Management
- Use clear, descriptive keys in the store
- Namespace keys to avoid conflicts (e.g., `weather_data` vs just `data`)
- Clean up temporary keys in the post() phase

### 2. Error Handling
- Always validate inputs in prep()
- Use try-except blocks in exec() for external calls
- Provide clear error messages in the store

### 3. Testing
- Write tests for all three phases of each node
- Test both success and failure paths
- Mock external services in tests

### 4. Documentation
- Add docstrings to all nodes and methods
- Document expected store keys
- Include usage examples

## AI-Assisted Development Workflow

### Using Cursor
1. Open the project in Cursor
2. The `.cursorrules` file will automatically guide AI suggestions
3. Use AI to generate nodes following the established patterns

### Using Claude-Code
```bash
# In your terminal
claude-code

# Ask Claude to create a new node
> Create a node that fetches user data from an API

# Claude will follow the patterns in CLAUDE.md
```

### Using Task-Master
1. Create your design document: `docs/design.md`
2. Run Task-Master to decompose into tasks
3. Work through tasks systematically

## Next Steps

1. Study the example nodes in `nodes.py`
2. Create your first custom node
3. Write tests for your node
4. Build a simple flow
5. Iterate and expand functionality

For more detailed API documentation, see [api-reference.md](api-reference.md).