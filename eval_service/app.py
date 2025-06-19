import os
import json
import argparse
import inspect
import traceback
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from transformers import HfArgumentParser
import uvicorn
from config import ServerConfig, ModelConfig, ToolConfig
from model_service import ModelService

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Change to INFO to see more logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_app(server_config: ServerConfig, model_config: ModelConfig, tool_config: ToolConfig) -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Args:
        server_config: Server configuration object
        model_config: Model configuration object
        tool_config: Tool configuration object
        
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="LLM Code Tool Service",
        description="Large language model code tool calling service compatible with OpenAI API",
        version="1.0.0"
    )
    
    # Add CORS middleware to allow cross-origin requests
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Set debug mode based on environment
    if hasattr(server_config, "environment") and server_config.environment == "development":
        app.debug = True
    
    # Initialize the model service
    model_service = ModelService(model_config, tool_config)
    model_service.load_model()
    
    # Store service in application state
    app.state.model_service = model_service
    
    # Add middleware for global exception handling
    @app.middleware("http")
    async def log_exceptions(request: Request, call_next):
        logger.info(f"Received request to {request.url.path}")
        try:
            response = await call_next(request)
            logger.info(f"Successfully processed request to {request.url.path}")
            return response
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Unhandled exception: {str(e)}\n{error_details}")
            raise
    
    @app.post("/completions")
    async def completions(request: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses with tool calling capabilities
        """
        try:
            request_body = await request.json()
            logger.info(f"Received completions request: {json.dumps(request_body)}")
            response = await app.state.model_service.completions_async(request_body)
            logger.info("Successfully processed completions request")
            return response
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in completions endpoint: {str(e)}\n{error_details}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.post("/chat/completions/legacy")
    async def chat_completions_legacy(request: Request):
        """
        Legacy chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses with tool calling capabilities.
        This is the legacy implementation that does not properly handle multi-turn conversations.
        """
        try:
            # Add more detailed logging around request body parsing
            logger.info("Attempting to parse request body")
            try:
                request_body = await request.json()
            except Exception as e:
                logger.error(f"Failed to parse request body as JSON: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {str(e)}")

            if not request_body:
                logger.error("Request body is empty or None")
                raise HTTPException(status_code=400, detail="Request body cannot be empty")

            print(f"\n[DEBUG] Raw request body received: {json.dumps(request_body, indent=2)}")
            print(f"[DEBUG] Request body keys: {list(request_body.keys())}")
            print(f"[DEBUG] Request body type: {type(request_body)}")
            
            # Check if extra_body is in the raw request
            if "extra_body" in request_body:
                print(f"[DEBUG] extra_body contents: {json.dumps(request_body['extra_body'], indent=2)}")
            else:
                print("[DEBUG] extra_body not found in request")
                # Check if it's nested somewhere else
                print(f"[DEBUG] Full request structure: {json.dumps(request_body, indent=2)}")

            logger.info(f"Received chat completions request: {json.dumps(request_body)}")
            
            # Validate required fields
            if "messages" not in request_body:
                logger.error("'messages' field missing from request body")
                raise HTTPException(status_code=400, detail="'messages' field is required")
                
            if "model" not in request_body:
                logger.error("'model' field missing from request body")
                raise HTTPException(status_code=400, detail="'model' field is required")

            response = await app.state.model_service.chat_completions_async(request_body)
            logger.info("Successfully processed chat completions request")
            return response
        except HTTPException:
            raise
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in chat completions endpoint: {str(e)}\n{error_details}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    @app.post("/chat/completions")
    async def chat_completions_multi_turn(request: Request):
        """
        Chat completion API endpoint compatible with OpenAI
        
        Processes chat messages and returns model-generated responses with tool calling capabilities.
        This implementation properly handles multi-turn conversations and tool interactions.
        """
        try:
            # Parse and validate request body
            try:
                request_body = await request.json()
            except Exception as e:
                logger.error(f"Failed to parse request body as JSON: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {str(e)}")

            if not request_body:
                logger.error("Request body is empty or None")
                raise HTTPException(status_code=400, detail="Request body cannot be empty")

            # Validate required fields
            if "messages" not in request_body:
                logger.error("'messages' field missing from request body")
                raise HTTPException(status_code=400, detail="'messages' field is required")
                
            if "model" not in request_body:
                logger.error("'model' field missing from request body")
                raise HTTPException(status_code=400, detail="'model' field is required")

            print(f"\n[DEBUG] Chat request received: {json.dumps(request_body, indent=2)}")
            
            response = await app.state.model_service.chat_completions_multi_turn_async(request_body)
            logger.info("Successfully processed chat completions request")
            return response
        except HTTPException:
            raise
        except Exception as e:
            error_details = traceback.format_exc()
            logger.error(f"Error in chat completions endpoint: {str(e)}\n{error_details}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint to verify service availability"""
        logger.info("Health check requested")
        return {"status": "healthy"}
    
    return app

async def main_async():
    # Set up command line argument parsing
    hf_parser = HfArgumentParser((ServerConfig, ModelConfig, ToolConfig))
    server_config, model_config, tool_config = hf_parser.parse_args_into_dataclasses()    
    tool_config.post_init()
    
    # Create and run the application
    app = create_app(server_config, model_config, tool_config)
    
    # Configure and start the server with enhanced logging
    config = uvicorn.Config(
        app, 
        host=server_config.host, 
        port=server_config.port, 
        log_level=server_config.log_level,  # Changed from "error" to "debug" for better visibility
        ws_max_queue=server_config.ws_max_queue, 
        workers=server_config.workers*model_config.num_models,
        access_log=True,
        timeout_keep_alive=server_config.timeout_keep_alive  # Added keep-alive timeout setting
    )
    server = uvicorn.Server(config)
    await server.serve()

def main():
    import asyncio
    asyncio.run(main_async())

if __name__ == "__main__":
    main()