"""
Retry wrapper - Generic decorator for retrying functions with exponential backoff.
Handles both synchronous and asynchronous functions.
"""
import asyncio
import functools
import time
from typing import Any, Callable, TypeVar, Union
import random

F = TypeVar('F', bound=Callable[..., Any])

def retry_wrap(
    max_attempts: int = 3,
    backoff: float = 1.0,
    exceptions: tuple = (Exception,),
    jitter: bool = True
) -> Callable[[F], F]:
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        backoff: Base backoff time in seconds
        exceptions: Tuple of exceptions to catch and retry
        jitter: Whether to add random jitter to backoff time
        
    Returns:
        Decorated function that will retry on failure
    """
    def decorator(func: F) -> F:
        if asyncio.iscoroutinefunction(func):
            # Async function
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return await func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            wait_time = backoff * (2 ** attempt)
                            if jitter:
                                wait_time *= (0.5 + random.random())
                            
                            print(f"Attempt {attempt + 1} failed: {e}")
                            print(f"Retrying in {wait_time:.1f} seconds...")
                            await asyncio.sleep(wait_time)
                        else:
                            print(f"All {max_attempts} attempts failed")
                            raise
                
                raise last_exception
            
            return async_wrapper
        else:
            # Sync function
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        if attempt < max_attempts - 1:
                            wait_time = backoff * (2 ** attempt)
                            if jitter:
                                wait_time *= (0.5 + random.random())
                            
                            print(f"Attempt {attempt + 1} failed: {e}")
                            print(f"Retrying in {wait_time:.1f} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"All {max_attempts} attempts failed")
                            raise
                
                raise last_exception
            
            return sync_wrapper
    
    return decorator

# Test functions
if __name__ == "__main__":
    # Test sync function
    counter = 0
    
    @retry_wrap(max_attempts=3, backoff=0.1)
    def flaky_function():
        global counter
        counter += 1
        if counter < 3:
            raise ValueError(f"Attempt {counter} failed")
        return "Success!"
    
    result = flaky_function()
    print(f"Result: {result}")
    
    # Test async function
    async def test_async():
        counter = 0
        
        @retry_wrap(max_attempts=3, backoff=0.1)
        async def async_flaky_function():
            nonlocal counter
            counter += 1
            if counter < 3:
                raise ValueError(f"Async attempt {counter} failed")
            return "Async success!"
        
        result = await async_flaky_function()
        print(f"Async result: {result}")
    
    asyncio.run(test_async()) 