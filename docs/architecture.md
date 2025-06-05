# System Architecture

## Overview

claude-pocketflow implements a node-based architecture for building agentic applications. The system is designed to be modular, testable, and AI-development friendly.

## Architecture Principles

### 1. Separation of Concerns
- **Nodes**: Business logic encapsulation
- **Flows**: Orchestration and state transitions
- **Utilities**: External service integrations
- **Store**: Shared state management

### 2. Three-Phase Node Lifecycle
Each node follows a consistent pattern:
```
prep() → exec() → post()
```
This ensures predictable behavior and makes testing straightforward.

### 3. Action-Based Flow Control
Flows use explicit actions to determine transitions:
- Each node returns a status/action
- Flows map actions to next nodes
- Clear paths for error handling

## Component Architecture

### Node Layer
```
┌─────────────────────────────────────┐
│           Node Interface            │
├─────────────────────────────────────┤
│  prep(store) → validate & prepare   │
│  exec(store) → perform logic        │
│  post(store) → cleanup & format     │
└─────────────────────────────────────┘
```

Nodes are stateless and communicate only through the shared store.

### Flow Layer
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Node A    │────▶│   Node B    │────▶│   Node C    │
└─────────────┘     └─────────────┘     └─────────────┘
       │                    │                    │
       └────────────────────┴────────────────────┘
                     Shared Store
```

Flows orchestrate node execution based on transition rules.

### Utility Layer
```
┌─────────────────────────────────────┐
│          External Services          │
├─────────────────────────────────────┤
│  - API Clients                      │
│  - Database Connections             │
│  - File System Operations           │
│  - Third-party Integrations         │
└─────────────────────────────────────┘
```

Utilities provide clean interfaces to external services.

## Data Flow

### Store Structure
The shared store is a dictionary that flows through all nodes:
```python
store = {
    # Input data
    "user_input": "...",
    
    # Processing data
    "intermediate_result": {...},
    
    # Output data
    "final_output": "...",
    
    # Control flow
    "status": "success",
    "next_action": "continue",
    
    # Error handling
    "error": None,
    "retry_count": 0
}
```

### State Transitions
```
[Initial Store] → [Node A prep] → [Node A exec] → [Node A post] →
[Updated Store] → [Flow Decision] → [Node B prep] → ...
```

## Error Handling Strategy

### Node-Level Errors
- Caught in exec() phase
- Stored in store["error"]
- Status set to error action

### Flow-Level Errors
- Dedicated error handling nodes
- Retry logic with exponential backoff
- Circuit breakers for external services

### Example Error Flow
```
Normal Path:    Start → Process → Complete
                  ↓
Error Path:     Error → Retry → Process
                  ↓
Final Error:    Log → Notify → Fail
```

## Scalability Considerations

### Horizontal Scaling
- Nodes are stateless and can run in parallel
- Store can be backed by distributed cache
- Flows can be distributed across workers

### Performance Optimization
- Lazy loading of utilities
- Connection pooling for external services
- Caching of expensive operations
- Async execution where beneficial

## Security Architecture

### Input Validation
- All inputs validated in prep() phase
- Type checking with Python type hints
- Schema validation for complex inputs

### Secret Management
- Environment variables for secrets
- Never store secrets in code
- Rotate credentials regularly

### Audit Trail
- All node executions logged
- Store snapshots for debugging
- Error tracking and monitoring

## Testing Architecture

### Unit Testing
- Test each node phase independently
- Mock external dependencies
- Verify store transformations

### Integration Testing
- Test complete flows
- Use test doubles for services
- Validate error paths

### Test Structure
```
tests/
├── test_nodes.py      # Node unit tests
├── test_flows.py      # Flow integration tests
├── test_utils.py      # Utility tests
└── conftest.py        # Shared fixtures
```

## Development Patterns

### Node Patterns
1. **Validator Node**: Ensures data integrity
2. **Transformer Node**: Modifies data format
3. **Aggregator Node**: Combines multiple sources
4. **Router Node**: Directs flow based on conditions

### Flow Patterns
1. **Sequential Flow**: A → B → C
2. **Conditional Flow**: A → (B or C) → D
3. **Loop Flow**: A → B → (back to A or continue)
4. **Parallel Flow**: A → (B and C) → D

## Monitoring and Observability

### Logging
- Structured logging with context
- Log level configuration
- Centralized log aggregation

### Metrics
- Node execution time
- Flow completion rate
- Error frequency
- External service latency

### Tracing
- Correlation IDs across nodes
- Flow execution visualization
- Performance bottleneck identification