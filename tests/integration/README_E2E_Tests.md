# End-to-End Integration Tests for OWUI Adaptive Memory Plugin

This document describes the comprehensive end-to-end integration tests that validate complete user workflows for the OWUI Adaptive Memory Plugin.

## Overview

The end-to-end tests simulate realistic user interactions with the memory system, testing the complete flow from user input through memory extraction, storage, and retrieval back to the user. These tests validate that the system works correctly in real-world scenarios.

## Test Structure

### Test Files

- `test_e2e_workflows.py` - Main end-to-end test suite
- `run_e2e_tests.py` - Test runner script for validation
- `README_E2E_Tests.md` - This documentation

### Test Classes

#### 1. TestCompleteWorkflows
Tests the complete memory lifecycle from user input to memory storage and retrieval.

**Key Tests:**
- `test_complete_memory_lifecycle` - Full workflow: inlet → extraction → storage → outlet → injection

**Validates:**
- User message processing through inlet
- Memory extraction from conversations
- Memory storage via API calls
- Memory injection in subsequent conversations
- Performance thresholds for each stage

#### 2. TestMultiTurnConversations
Tests memory accumulation and context building across multiple conversation turns.

**Key Tests:**
- `test_multi_turn_memory_accumulation` - Progressive memory building
- `test_conversation_context_evolution` - How context evolves over time

**Validates:**
- Memory persistence across conversation turns
- Context injection based on conversation history
- Evolving user preferences and information

#### 3. TestRealisticConversationScenarios
Tests realistic conversation patterns that users would have with the system.

**Key Tests:**
- `test_new_user_onboarding_conversation` - First-time user experience
- `test_technical_support_conversation` - Technical help with context
- `test_shopping_preferences_conversation` - Preference-based recommendations
- `test_multi_language_conversation` - Multi-language support

**Validates:**
- Natural conversation flows
- Context building in domain-specific conversations
- Memory extraction from specialized topics
- Cross-language memory persistence

#### 4. TestMemoryDeduplicationAndOrchestration
Tests the memory deduplication system and filter orchestration during real workflows.

**Key Tests:**
- `test_memory_deduplication_workflow` - Duplicate information handling
- `test_filter_orchestration_workflow` - Multi-filter coordination

**Validates:**
- Duplicate memory detection and merging
- Filter coordination and conflict resolution
- Performance impact of orchestration system

#### 5. TestPerformanceAndReliability
Tests system performance under realistic load and error recovery capabilities.

**Key Tests:**
- `test_concurrent_conversations_performance` - Multiple simultaneous users
- `test_memory_under_load` - Memory usage under high load
- `test_error_recovery_workflow` - Error handling and recovery

**Validates:**
- Concurrent user handling
- Memory leak prevention
- Error recovery without data loss
- Performance thresholds under load

#### 6. TestCrossSessionPersistence
Tests memory persistence across different user sessions.

**Key Tests:**
- `test_cross_session_memory_persistence` - Long-term memory retention
- `test_user_isolation_across_sessions` - User data isolation

**Validates:**
- Memory persistence between sessions
- User data isolation and privacy
- Long-term context availability

## Test Scenarios

### Complete Memory Workflow

```python
# User message → inlet processing → memory extraction → API storage
user_input = "Hi, I'm Alex and I work as a data scientist at Netflix."

# System processes through inlet
processed_inlet = await filter.async_inlet(conversation, events, user)

# LLM generates response
llm_response = "Nice to meet you Alex! Netflix must have fascinating data challenges."

# System processes through outlet (extracts memories)
processed_outlet = await filter.async_outlet(full_conversation, events, user)

# Later conversation → memory injection
new_conversation = "Can you recommend Python libraries for data visualization?"
processed_new = await filter.async_inlet(new_conversation, events, user)
# Should inject context about user being a data scientist
```

### Multi-Turn Conversation

```python
# Progressive conversation building context
turns = [
    "Hi, I'm Sarah, I'm 28 and work as a software engineer.",
    "I mainly work with Python and PostgreSQL.",
    "I also enjoy hiking on weekends.",
    "Can you recommend Python libraries for database optimization?"
]

# Each turn builds on previous context
# Final turn should have context of: name, age, job, technologies, hobbies
```

### Realistic Scenarios

#### New User Onboarding
- Initial introduction and preference gathering
- Progressive information collection
- Context building for personalized responses

#### Technical Support
- Problem description with technical details
- Solution suggestions based on user context
- Follow-up questions referencing previous context

#### Shopping Preferences
- Initial requirements gathering
- Preference refinement based on additional information
- Recommendations based on accumulated preferences

#### Multi-Language Support
- Conversations in multiple languages
- Memory persistence across language switches
- Context injection in appropriate languages

## Performance Requirements

### Response Time Thresholds
- Inlet processing: < 10 seconds
- Outlet processing: < 10 seconds
- Memory injection: < 5 seconds
- Context retrieval: < 2 seconds

### Throughput Requirements
- Concurrent users: ≥ 5 simultaneous conversations
- Messages per second: ≥ 10 msg/s under load
- Memory operations: ≥ 100 operations/minute

### Resource Usage
- Memory growth: < 500MB per 50 messages
- CPU usage: < 80% during peak load
- Error rate: < 5% under normal conditions

## Error Handling

### Test Scenarios
- Empty messages
- Extremely long messages (10,000+ characters)
- Messages with special characters
- Malformed JSON-like content
- SQL injection attempts
- XSS attempts

### Expected Behavior
- Graceful degradation without crashes
- Error logging without data exposure
- Fallback to basic functionality
- User notification of processing issues

## Running the Tests

### Full Test Suite
```bash
# Run all end-to-end tests
python -m pytest tests/integration/test_e2e_workflows.py -v

# Run with detailed output
python -m pytest tests/integration/test_e2e_workflows.py -v -s
```

### Specific Test Categories
```bash
# Complete workflows only
python -m pytest tests/integration/test_e2e_workflows.py::TestCompleteWorkflows -v

# Performance tests only
python -m pytest tests/integration/test_e2e_workflows.py::TestPerformanceAndReliability -v

# Realistic scenarios only
python -m pytest tests/integration/test_e2e_workflows.py::TestRealisticConversationScenarios -v
```

### Test Runner Script
```bash
# Quick validation of test infrastructure
python3 run_e2e_tests.py
```

### With Mock Infrastructure
```bash
# Run with mock backends (when real APIs unavailable)
python -m pytest tests/integration/test_e2e_workflows.py -v --mock-all
```

## Test Data

### User Profiles
- Software engineers
- Data scientists  
- Marketing professionals
- Students
- Multi-language users

### Conversation Types
- Technical discussions
- Personal preferences
- Shopping decisions
- Learning and education
- Problem-solving

### Memory Categories
- Personal information (name, job, location)
- Technical preferences (languages, tools)
- Behavioral patterns (hobbies, interests)
- Professional context (company, role, experience)
- Learning goals and progress

## Integration with OpenWebUI

### API Endpoints Tested
- Memory storage: `POST /api/v1/memories`
- Memory retrieval: `GET /api/v1/memories?user_id={id}`
- Memory search: `GET /api/v1/memories/search?query={query}`
- Memory updates: `PUT /api/v1/memories/{id}`
- Memory deletion: `DELETE /api/v1/memories/{id}`

### Filter Interface
- `inlet()` method for preprocessing user input
- `outlet()` method for processing LLM responses
- Event emission for status updates
- Error handling and fallback mechanisms

### Mock Infrastructure
- OpenWebUI API mock server
- LLM API mock responses
- WebSocket mock for real-time events
- Database mock for memory persistence

## Monitoring and Metrics

### Performance Metrics
- Response times for each operation
- Memory usage during processing
- CPU utilization under load
- Error rates by operation type

### Quality Metrics
- Memory extraction accuracy
- Context injection relevance
- User preference persistence
- Cross-session continuity

### Business Metrics
- User engagement improvement
- Conversation quality scores
- Memory utilization rates
- System adoption metrics

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Fix Python path issues
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

#### Mock Filter Fallback
If the real filter fails to import, tests automatically fall back to a mock implementation for infrastructure testing.

#### Performance Test Failures
Adjust thresholds in test configuration if running on slower hardware:

```python
# In test_config.py
PERFORMANCE_THRESHOLDS = {
    "memory_extraction": {
        "max_duration_ms": 1000,  # Increased from 500
        "max_memory_mb": 100      # Increased from 50
    }
}
```

### Debug Mode
```bash
# Run with debug logging
python -m pytest tests/integration/test_e2e_workflows.py -v -s --log-cli-level=DEBUG
```

### Test Coverage
```bash
# Generate coverage report
python -m pytest tests/integration/test_e2e_workflows.py --cov=adaptive_memory_v4_0 --cov-report=html
```

## Contributing

### Adding New Test Scenarios
1. Create new test methods in appropriate test classes
2. Follow naming convention: `test_scenario_description`
3. Include performance validation
4. Add error handling verification
5. Update this documentation

### Test Data Generation
Use the provided helper functions:
- `generate_test_user()` - Create test users
- `generate_test_memories()` - Create test memories  
- `generate_test_messages()` - Create test conversations

### Performance Benchmarking
Include performance metrics in all tests:
```python
performance_monitor.record_metric("operation_time", duration)
assert duration < threshold, f"Operation too slow: {duration}s"
```

## Conclusion

These end-to-end tests provide comprehensive validation of the OWUI Adaptive Memory Plugin's core functionality, ensuring that the system works correctly in realistic user scenarios while maintaining performance and reliability standards.

The tests serve as both validation tools and documentation of expected system behavior, helping developers understand how the memory system should function in practice.