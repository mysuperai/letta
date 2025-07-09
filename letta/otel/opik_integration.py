"""
Comprehensive Opik integration for Letta - LLM tracing and Agent tracing
"""

import os
import time
import uuid
from contextlib import asynccontextmanager
from functools import wraps
from typing import Any, Dict, List, Optional

from letta.log import get_logger

logger = get_logger(__name__)

# Global flag to track initialization
_opik_initialized = False


# =============================================================================
# Core Opik Setup and Configuration
# =============================================================================


def check_opik_environment():
    """
    Check if Opik environment variables are configured
    
    Returns:
        bool: True if Opik is configured, False otherwise
    """
    # Check if tracking is explicitly disabled
    if os.environ.get("OPIK_TRACK_DISABLE", "false").lower() == "true":
        logger.info("Opik tracking is disabled via OPIK_TRACK_DISABLE")
        return False
    
    # Check for required environment variables
    opik_url = os.environ.get("OPIK_URL_OVERRIDE")
    
    if not opik_url:
        logger.debug("OPIK_URL_OVERRIDE not set, Opik tracing will be disabled")
        return False
    
    logger.info(f"Opik environment detected: {opik_url}")
    return True


def setup_opik_tracing():
    """
    Set up Opik tracing for Letta using environment variables
    
    Follows standard Opik SDK configuration pattern:
    - OPIK_URL_OVERRIDE: Opik server URL
    - OPIK_API_KEY: API key (optional for self-hosted)
    - OPIK_WORKSPACE: Workspace name (optional for self-hosted)
    - OPIK_PROJECT_NAME: Project name (optional)
    - OPIK_TRACK_DISABLE: Disable tracking (default: false)
    """
    global _opik_initialized

    if _opik_initialized:
        logger.debug("Opik tracing already initialized")
        return

    # Check if Opik environment is configured
    if not check_opik_environment():
        logger.debug("Opik environment not configured, skipping initialization")
        return

    try:
        # Import Opik SDK
        import opik
        
        # Configure Opik using environment variables
        opik.configure()
        
        _opik_initialized = True
        
        opik_url = os.environ.get("OPIK_URL_OVERRIDE")
        project_name = os.environ.get("OPIK_PROJECT_NAME", "letta-traces")
        
        logger.info(f"Opik tracing initialized successfully")
        logger.info(f"  URL: {opik_url}")
        logger.info(f"  Project: {project_name}")

    except ImportError as e:
        logger.warning(f"Opik SDK not installed: {e}")
        logger.info("Continuing without Opik tracing")
        _opik_initialized = False
    except Exception as e:
        logger.warning(f"Failed to initialize Opik tracing: {e}")
        logger.info("Continuing without Opik tracing")
        _opik_initialized = False


def is_opik_enabled() -> bool:
    """Check if Opik tracing is enabled and initialized"""
    return _opik_initialized


# =============================================================================
# LLM Client Tracking
# =============================================================================


def track_llm_call(func):
    """
    Decorator to track LLM calls with Opik
    """

    def wrapper(*args, **kwargs):
        if not _opik_initialized:
            # Fallback to regular function call if Opik not initialized
            return func(*args, **kwargs)

        try:
            from opik import track

            # Create a tracked version of the function
            tracked_func = track(func)
            return tracked_func(*args, **kwargs)

        except ImportError:
            logger.warning("Opik SDK not available, falling back to untracked call")
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in Opik tracking: {e}")
            return func(*args, **kwargs)

    return wrapper


def track_openai_client(openai_client):
    """
    Wrap OpenAI client with Opik tracking

    Args:
        openai_client: OpenAI client instance

    Returns:
        Tracked OpenAI client or original client if tracking fails
    """
    if not _opik_initialized:
        return openai_client

    try:
        from opik.integrations.openai import track_openai

        return track_openai(openai_client)
    except ImportError:
        logger.warning("Opik SDK not available, using untracked OpenAI client")
        return openai_client
    except Exception as e:
        logger.error(f"Error tracking OpenAI client: {e}")
        return openai_client


def track_anthropic_client(anthropic_client):
    """
    Wrap Anthropic client with Opik tracking

    Args:
        anthropic_client: Anthropic client instance

    Returns:
        Tracked Anthropic client or original client if tracking fails
    """
    if not _opik_initialized:
        return anthropic_client

    try:
        from opik.integrations.anthropic import track_anthropic

        return track_anthropic(anthropic_client)
    except ImportError:
        logger.warning("Opik Anthropic integration not available, using untracked client")
        return anthropic_client
    except Exception as e:
        logger.error(f"Error tracking Anthropic client: {e}")
        return anthropic_client


def log_llm_trace(
    model: str,
    input_text: str,
    output_text: str,
    token_count: Optional[int] = None,
    latency: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log LLM trace data to Opik

    Args:
        model: Model name
        input_text: Input prompt
        output_text: Generated output
        token_count: Optional token count
        latency: Optional response latency
        metadata: Optional additional metadata
    """
    if not _opik_initialized:
        return

    try:
        from opik import track

        # Create a trace with LLM call details
        trace_data = {"model": model, "input": input_text, "output": output_text, "timestamp": time.time()}

        if token_count is not None:
            trace_data["token_count"] = token_count
        if latency is not None:
            trace_data["latency"] = latency
        if metadata:
            trace_data.update(metadata)

        # Log the trace (this would be called within a tracked function)
        logger.debug(f"Logging LLM trace: {trace_data}")

    except ImportError:
        logger.warning("Opik SDK not available for trace logging")
    except Exception as e:
        logger.error(f"Error logging LLM trace: {e}")


# =============================================================================
# Agent Tracing System
# =============================================================================


class AgentTracer:
    """Agent tracing utilities for Letta"""

    def __init__(self):
        self.active_traces = {}
        self.conversation_groups = {}

    def create_conversation_group(self, agent_id: str, user_id: str = None) -> str:
        """Create a conversation group ID for tracking multi-turn conversations"""
        group_id = f"conversation-{agent_id}-{uuid.uuid4().hex[:8]}"
        self.conversation_groups[group_id] = {"agent_id": agent_id, "user_id": user_id, "created_at": time.time(), "message_count": 0}
        return group_id

    def track_agent_step(self, agent_id: str, step_count: int, metadata: Dict[str, Any] = None):
        """Track an agent step execution"""
        if not is_opik_enabled():
            return lambda func: func

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    from opik import track

                    # Create span name with agent and step info
                    span_name = f"agent_step_{agent_id}_{step_count}"

                    # Build metadata
                    trace_metadata = {"agent_id": agent_id, "step_count": step_count, "timestamp": time.time(), "operation": "agent_step"}
                    if metadata:
                        trace_metadata.update(metadata)

                    # Execute with tracking
                    with track(name=span_name, metadata=trace_metadata):
                        result = await func(*args, **kwargs)
                        return result

                except ImportError:
                    logger.debug("Opik not available, executing without agent tracing")
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in agent step tracing: {e}")
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    from opik import track

                    span_name = f"agent_step_{agent_id}_{step_count}"
                    trace_metadata = {"agent_id": agent_id, "step_count": step_count, "timestamp": time.time(), "operation": "agent_step"}
                    if metadata:
                        trace_metadata.update(metadata)

                    with track(name=span_name, metadata=trace_metadata):
                        result = func(*args, **kwargs)
                        return result

                except ImportError:
                    logger.debug("Opik not available, executing without agent tracing")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in agent step tracing: {e}")
                    return func(*args, **kwargs)

            return async_wrapper if hasattr(func, "__code__") and func.__code__.co_flags & 0x0080 else sync_wrapper

        return decorator

    def track_tool_execution(self, tool_name: str = None, agent_id: str = None, metadata: Dict[str, Any] = None):
        """Track tool execution within an agent"""
        if not is_opik_enabled():
            return lambda func: func

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    from opik import track

                    # Extract tool info from method arguments if not provided
                    actual_tool_name = tool_name
                    actual_agent_id = agent_id

                    # For ToolExecutionManager, extract from arguments
                    if not actual_tool_name and len(args) >= 3:
                        # args[1] is function_name, args[3] is tool object
                        actual_tool_name = args[1] if len(args) > 1 else "unknown"
                        if len(args) > 3 and hasattr(args[3], "name"):
                            actual_tool_name = args[3].name

                    # Try to get agent_id from self
                    if not actual_agent_id and hasattr(args[0], "agent_state") and hasattr(args[0].agent_state, "id"):
                        actual_agent_id = args[0].agent_state.id

                    span_name = f"tool_{actual_tool_name or 'unknown'}"
                    trace_metadata = {
                        "tool_name": actual_tool_name or "unknown",
                        "agent_id": actual_agent_id or "unknown",
                        "timestamp": time.time(),
                        "operation": "tool_execution",
                    }
                    if metadata:
                        trace_metadata.update(metadata)

                    # Add function arguments to trace
                    if len(args) >= 3:
                        trace_metadata["function_args"] = str(args[2])  # function_args

                    with track(name=span_name, metadata=trace_metadata):
                        result = await func(*args, **kwargs)

                        # Add result info to trace
                        if hasattr(result, "status"):
                            trace_metadata["execution_status"] = result.status
                        if hasattr(result, "func_return"):
                            trace_metadata["function_return"] = str(result.func_return)[:500]  # Truncate for readability

                        return result

                except ImportError:
                    logger.debug("Opik not available, executing without tool tracing")
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in tool execution tracing: {e}")
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    from opik import track

                    # Extract tool info from method arguments if not provided
                    actual_tool_name = tool_name
                    actual_agent_id = agent_id

                    if not actual_tool_name and len(args) >= 3:
                        actual_tool_name = args[1] if len(args) > 1 else "unknown"
                        if len(args) > 3 and hasattr(args[3], "name"):
                            actual_tool_name = args[3].name

                    if not actual_agent_id and hasattr(args[0], "agent_state") and hasattr(args[0].agent_state, "id"):
                        actual_agent_id = args[0].agent_state.id

                    span_name = f"tool_{actual_tool_name or 'unknown'}"
                    trace_metadata = {
                        "tool_name": actual_tool_name or "unknown",
                        "agent_id": actual_agent_id or "unknown",
                        "timestamp": time.time(),
                        "operation": "tool_execution",
                    }
                    if metadata:
                        trace_metadata.update(metadata)

                    if len(args) >= 3:
                        trace_metadata["function_args"] = str(args[2])

                    with track(name=span_name, metadata=trace_metadata):
                        result = func(*args, **kwargs)

                        if hasattr(result, "status"):
                            trace_metadata["execution_status"] = result.status
                        if hasattr(result, "func_return"):
                            trace_metadata["function_return"] = str(result.func_return)[:500]

                        return result

                except ImportError:
                    logger.debug("Opik not available, executing without tool tracing")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in tool execution tracing: {e}")
                    return func(*args, **kwargs)

            return async_wrapper if hasattr(func, "__code__") and func.__code__.co_flags & 0x0080 else sync_wrapper

        return decorator

    @asynccontextmanager
    async def track_conversation(self, agent_id: str, user_message: str, group_id: str = None):
        """Context manager for tracking entire conversations"""
        if not is_opik_enabled():
            yield group_id
            return

        try:
            from opik import track

            if not group_id:
                group_id = self.create_conversation_group(agent_id)

            # Update conversation metadata
            if group_id in self.conversation_groups:
                self.conversation_groups[group_id]["message_count"] += 1

            span_name = f"conversation_{agent_id}"
            trace_metadata = {
                "agent_id": agent_id,
                "group_id": group_id,
                "user_message": user_message,
                "timestamp": time.time(),
                "operation": "conversation",
            }

            with track(name=span_name, metadata=trace_metadata):
                yield group_id

        except ImportError:
            logger.debug("Opik not available, executing without conversation tracing")
            yield group_id
        except Exception as e:
            logger.error(f"Error in conversation tracing: {e}")
            yield group_id

    def track_multi_agent_interaction(self, agents: List[str], interaction_type: str, metadata: Dict[str, Any] = None):
        """Track multi-agent interactions"""
        if not is_opik_enabled():
            return lambda func: func

        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    from opik import track

                    span_name = f"multi_agent_{interaction_type}"
                    trace_metadata = {
                        "agents": agents,
                        "interaction_type": interaction_type,
                        "timestamp": time.time(),
                        "operation": "multi_agent_interaction",
                    }
                    if metadata:
                        trace_metadata.update(metadata)

                    with track(name=span_name, metadata=trace_metadata):
                        result = await func(*args, **kwargs)
                        return result

                except ImportError:
                    logger.debug("Opik not available, executing without multi-agent tracing")
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in multi-agent tracing: {e}")
                    return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                try:
                    from opik import track

                    span_name = f"multi_agent_{interaction_type}"
                    trace_metadata = {
                        "agents": agents,
                        "interaction_type": interaction_type,
                        "timestamp": time.time(),
                        "operation": "multi_agent_interaction",
                    }
                    if metadata:
                        trace_metadata.update(metadata)

                    with track(name=span_name, metadata=trace_metadata):
                        result = func(*args, **kwargs)
                        return result

                except ImportError:
                    logger.debug("Opik not available, executing without multi-agent tracing")
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in multi-agent tracing: {e}")
                    return func(*args, **kwargs)

            return async_wrapper if hasattr(func, "__code__") and func.__code__.co_flags & 0x0080 else sync_wrapper

        return decorator


# =============================================================================
# Global Instance and Convenience Functions
# =============================================================================

# Global agent tracer instance
agent_tracer = AgentTracer()


def track_agent_step(agent_id: str, step_count: int, metadata: Dict[str, Any] = None):
    """Decorator for tracking agent steps"""
    return agent_tracer.track_agent_step(agent_id, step_count, metadata)


def track_tool_execution(tool_name: str = None, agent_id: str = None, metadata: Dict[str, Any] = None):
    """Decorator for tracking tool executions"""
    return agent_tracer.track_tool_execution(tool_name, agent_id, metadata)


def track_multi_agent_interaction(agents: List[str], interaction_type: str, metadata: Dict[str, Any] = None):
    """Decorator for tracking multi-agent interactions"""
    return agent_tracer.track_multi_agent_interaction(agents, interaction_type, metadata)


def track_conversation(agent_id: str, user_message: str, group_id: str = None):
    """Context manager for tracking conversations"""
    return agent_tracer.track_conversation(agent_id, user_message, group_id)


def log_agent_event(agent_id: str, event_type: str, data: Dict[str, Any] = None):
    """Log agent events for debugging and monitoring"""
    if not is_opik_enabled():
        return

    try:
        from opik import track

        event_data = {"agent_id": agent_id, "event_type": event_type, "timestamp": time.time(), "operation": "agent_event"}
        if data:
            event_data.update(data)

        logger.info(f"Agent event: {event_type} for agent {agent_id}")

    except ImportError:
        logger.debug("Opik not available for agent event logging")
    except Exception as e:
        logger.error(f"Error logging agent event: {e}")
