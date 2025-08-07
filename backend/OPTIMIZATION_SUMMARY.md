# OpenAI API Call Optimization

## Problem
The backend project was making **two separate OpenAI chat completion calls** for each query:

1. **First call** in `RetrievalService.search()` - Used LlamaIndex's query engine which internally called the `DeepInfraLLM.complete()` method
2. **Second call** in `LLMOrchestrator.generate()` - Made a direct OpenAI client call to generate the final response

This resulted in:
- **Increased latency** (2x API call overhead)
- **Higher costs** (paying for two separate API calls)
- **Redundant processing** (similar context being processed twice)

## Solution
Optimized the RAG pipeline to use **only one OpenAI API call**:

### Changes Made

#### 1. Modified RetrievalService (`app/core/service.py`)
- **Before**: Used `index.as_query_engine()` which internally makes LLM calls
- **After**: Changed to use `index.as_retriever()` which only retrieves documents without LLM generation
- **Return type**: Changed from `QueryResponse` to `List[Document]`
- **Removed**: Unused DeepInfraLLM dependency from RetrievalService

#### 2. Updated RAGWorkflowOrchestrator (`app/core/workflow.py`)
- **Before**: Retrieved `QueryResponse` from RetrievalService, then extracted source documents
- **After**: Directly receives `List[Document]` from RetrievalService
- **Flow**: Now does pure retrieval first, then single LLM generation

#### 3. Maintained LLMOrchestrator (`app/core/orchestrator.py`)
- **No changes needed** - Still handles the single LLM call for response generation
- **Input**: Receives query and source documents
- **Output**: Generates final response using context

### New Flow
```
Query → RetrievalService.search() → [Documents] → LLMOrchestrator.generate() → Response
         (Vector similarity only)                    (Single OpenAI API call)
```

### Previous Flow
```
Query → RetrievalService.search() → QueryResponse → LLMOrchestrator.generate() → Response
         (OpenAI API call #1)                        (OpenAI API call #2)
```

## Benefits
- **50% reduction in API calls** (2 → 1)
- **Improved latency** (single network round-trip instead of two)
- **Cost savings** (half the API usage)
- **Cleaner separation of concerns** (retrieval vs generation)
- **Better error handling** (single point of failure for LLM calls)

## Backward Compatibility
- ✅ **API endpoints unchanged** - `/answer` endpoint works exactly the same
- ✅ **Response format unchanged** - Same JSON structure returned
- ✅ **Configuration unchanged** - Same environment variables and settings
- ✅ **Caching unchanged** - Document processing cache still works

## Files Modified
1. `app/core/service.py` - Optimized RetrievalService
2. `app/core/workflow.py` - Updated workflow orchestration
3. Removed unused imports and dependencies

The optimization maintains all existing functionality while significantly improving performance and reducing costs.
