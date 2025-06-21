import pickle
import requests
from typing import Any, Dict, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FlaskWrapper:
    """
    Wrapper class for code2seq Flask API communication
    Provides health checking, server information, and main processing functionality
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:5001", timeout: int = 30) -> None:
        self.base_url = base_url
        self.api_url = f"{self.base_url}/api/code2seq"
        self.timeout = timeout
        
    def get(self, source_code: str) -> Any:
        """
        Main processing: send source code and get sequences
        
        Args:
            source_code: Java source code to process
            
        Returns:
            Processed result from code2seq model
            
        Raises:
            Exception: If server returns error or communication fails
        """
        if not source_code or not source_code.strip():
            raise ValueError("Source code cannot be empty")
            
        data = {"source_code": source_code}
        
        try:
            response = requests.post(
                self.api_url, 
                json=data, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"Server error: {response.status_code}")
                raise Exception(f"Server error: {response.status_code}")
            
            data_binary = response.content
            return pickle.loads(data_binary)
            
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout after {self.timeout} seconds")
            raise Exception(f"Request timeout after {self.timeout} seconds")
        except requests.exceptions.ConnectionError:
            logger.error("Connection error - server may be down")
            raise Exception("Connection error - server may be down")

    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            response = requests.get(
                f"{self.base_url}/health", 
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

    def server_info(self) -> Dict[str, Any]:
        """Get detailed server information"""
        try:
            response = requests.get(self.base_url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Server info request failed: {e}")
            return {"status": "error", "message": str(e)}

    def is_healthy(self) -> bool:
        """Quick health status check"""
        try:
            health = self.health_check()
            return health.get("status") == "healthy"
        except:
            return False
            
    def test_connection(self) -> Dict[str, Any]:
        """Test basic connectivity and functionality"""
        result = {
            "connectivity": False,
            "health_status": False,
            "processing_test": False,
            "details": {}
        }
        
        try:
            # Test basic connectivity
            info = self.server_info()
            if info.get("status") != "error":
                result["connectivity"] = True
                result["details"]["server_info"] = info
            
            # Test health endpoint
            health = self.health_check()
            if health.get("status") == "healthy":
                result["health_status"] = True
                result["details"]["health"] = health
            
            # Test actual processing
            if result["health_status"]:
                test_code = "public class Test { int getValue() { return 42; } }"
                process_result = self.get(test_code)
                result["processing_test"] = True
                result["details"]["test_result_length"] = len(process_result) if process_result else 0
                
        except Exception as e:
            result["details"]["error"] = str(e)
            
        return result


if __name__ == "__main__":
    wrapper = FlaskWrapper()
    
    print("=== Code2Seq Flask Wrapper Test ===")
    
    # Comprehensive connection test
    print("\n=== Connection Test ===")
    test_result = wrapper.test_connection()
    for key, value in test_result.items():
        print(f"{key}: {value}")
    
    # Processing test if server is healthy
    if test_result["processing_test"]:
        print("\n=== Processing Test ===")
        try:
            test_code = """
            public class Calculator {
                public int add(int a, int b) {
                    return a + b;
                }
                
                public int multiply(int x, int y) {
                    return x * y;
                }
            }
            """
            result = wrapper.get(test_code)
            print(f"Processing successful: {len(result)} items returned")
        except Exception as e:
            print(f"Processing failed: {e}")
    else:
        print("Server not ready for processing tests")
